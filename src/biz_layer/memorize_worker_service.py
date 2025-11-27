"""
记忆提取 Worker 服务

提供全局队列和常驻 Worker，用于异步处理 MemCell 的记忆提取任务。
"""

import asyncio
import traceback
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from memory_layer.memory_manager import MemoryManager
from api_specs.dtos.memory_command import MemorizeRequest
from api_specs.memory_types import (
    MemoryType,
    MemCell,
    Memory,
    SemanticMemoryItem,
)
from memory_layer.memory_extractor.event_log_extractor import EventLog
from core.observation.logger import get_logger
from core.di import get_bean_by_type
from core.di.decorators import service
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from biz_layer.mem_db_operations import (
    _convert_episode_memory_to_doc,
    _convert_semantic_memory_to_doc,
    _convert_event_log_to_docs,
)
from biz_layer.mem_memorize import (
    MemoryDocPayload,
    save_memory_docs,
    load_core_memories,
    update_status_after_memcell,
    if_memorize,
    _trigger_clustering,
)

logger = get_logger(__name__)


class RequestStatus(str, Enum):
    """请求状态"""
    PENDING = "pending"       # 已提交，等待处理
    PROCESSING = "processing" # 正在处理
    COMPLETED = "completed"   # 处理完成
    FAILED = "failed"         # 处理失败


@service(name="memorize_worker_service", primary=True)
class MemorizeWorkerService:
    """
    记忆提取 Worker 服务
    
    管理全局队列和常驻 Worker，提供异步的 MemCell 处理能力。
    """
    
    def __init__(self):
        """初始化 Worker 服务"""
        self.memcell_queue: asyncio.Queue = asyncio.Queue()
        self.worker: Optional[asyncio.Task] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._request_status: Dict[str, RequestStatus] = {}  # request_id -> status
        
        logger.info(f"[MemorizeWorkerService] 初始化")
    
    async def start(self):
        """启动 Worker"""
        if self._running:
            logger.warning("[MemorizeWorkerService] Worker 已经在运行中")
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        self.worker = asyncio.create_task(self._worker())
        logger.info(f"[MemorizeWorkerService] Worker 已启动")
    
    async def stop(self, timeout: float = 30.0):
        """
        停止 Worker
        
        Args:
            timeout: 等待超时时间（秒）
        """
        if not self._running:
            logger.warning("[MemorizeWorkerService] Worker 未运行")
            return
        
        logger.info(f"[MemorizeWorkerService] 停止 Worker...")
        self._running = False
        
        # 发送结束信号
        await self.memcell_queue.put(None)
        
        # 设置 shutdown 事件
        self._shutdown_event.set()
        
        # 等待 Worker 完成
        if self.worker:
            try:
                await asyncio.wait_for(self.worker, timeout=timeout)
                logger.info("[MemorizeWorkerService] Worker 已停止")
            except asyncio.TimeoutError:
                logger.warning(f"[MemorizeWorkerService] Worker 停止超时（{timeout}秒），强制取消")
                if not self.worker.done():
                    self.worker.cancel()
        
        self.worker = None
    
    async def submit_memcell(
        self,
        memcell: MemCell,
        request: MemorizeRequest,
        current_time: datetime,
    ) -> str:
        """
        提交 MemCell 到队列进行异步处理
        
        Args:
            memcell: 要处理的 MemCell
            request: 原始请求
            current_time: 当前时间
            
        Returns:
            request_id: 用于追踪请求状态
        """
        if not self._running:
            await self.start()
        
        # 生成 request_id（用 memcell.event_id）
        request_id = memcell.event_id
        self._request_status[request_id] = RequestStatus.PENDING
        
        # 将任务放入队列
        task_data = {
            'request_id': request_id,
            'memcell': memcell,
            'request': request,
            'current_time': current_time,
        }
        await self.memcell_queue.put(task_data)
        logger.info(f"[MemorizeWorkerService] MemCell 已提交到队列: {request_id}")
        return request_id
    
    def get_status(self, request_id: str) -> Optional[RequestStatus]:
        """查询请求状态"""
        return self._request_status.get(request_id)
    
    def is_completed(self, request_id: str) -> bool:
        """检查请求是否完成"""
        status = self._request_status.get(request_id)
        return status in (RequestStatus.COMPLETED, RequestStatus.FAILED)
    
    async def _worker(self):
        """Worker 主循环"""
        logger.info(f"[Worker] 启动")
        memory_manager = MemoryManager()
        
        while self._running or not self.memcell_queue.empty():
            try:
                # 从队列获取任务（带超时，避免永久阻塞）
                try:
                    task_data = await asyncio.wait_for(
                        self.memcell_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 收到结束信号
                if task_data is None:
                    logger.info(f"[Worker] 收到结束信号")
                    self.memcell_queue.task_done()
                    break
                
                request_id = task_data['request_id']
                memcell = task_data['memcell']
                request = task_data['request']
                current_time = task_data['current_time']
                
                # 更新状态为处理中
                self._request_status[request_id] = RequestStatus.PROCESSING
                logger.info(f"[Worker] 开始处理 MemCell: {request_id}")
                
                try:
                    # 处理 MemCell
                    await self._process_memcell(
                        memcell=memcell,
                        request=request,
                        memory_manager=memory_manager,
                        current_time=current_time,
                    )
                    # 更新状态为完成
                    self._request_status[request_id] = RequestStatus.COMPLETED
                    logger.info(f"[Worker] 完成处理 MemCell: {request_id}")
                except Exception as e:
                    # 更新状态为失败
                    self._request_status[request_id] = RequestStatus.FAILED
                    logger.error(f"[Worker] 处理 MemCell 失败: {request_id}, error={e}")
                    traceback.print_exc()
                
                self.memcell_queue.task_done()
                
            except Exception as e:
                logger.error(f"[Worker] 处理任务异常: {e}")
                traceback.print_exc()
                try:
                    self.memcell_queue.task_done()
                except ValueError:
                    pass
        
        logger.info(f"[Worker] 退出")

    def _clone_memory_for_user(self, memory: Memory, user_id: str) -> Memory:
        """复制 Memory 并设置新的 user_id"""
        from dataclasses import replace
        return replace(memory, user_id=user_id, user_name=user_id)

    async def _process_memcell(
        self,
        memcell: MemCell,
        request: MemorizeRequest,
        memory_manager: MemoryManager,
        current_time: datetime,
    ):
        """
        处理单个 MemCell 的所有记忆提取任务
        
        包括:
        1. 提取群组和个人 Episode
        2. 更新 MemCell 的 episode 字段到数据库
        3. 触发聚类 (异步，不等待)
        4. 提取 Semantic 和 EventLog
        5. 保存所有记忆到数据库
        """
        # ===== 第一步：获取场景信息并判断是否只提取群组 Episode =====
        conversation_meta_repo = get_bean_by_type(ConversationMetaRawRepository)
        conversation_meta = await conversation_meta_repo.get_by_group_id(request.group_id)
        scene = conversation_meta.scene if conversation_meta and conversation_meta.scene else "assistant"
        is_assistant_scene = scene.lower() in ["assistant", "companion"]
        
        # ===== 第二步：提取所有参与者 =====
        participants = []
        if memcell.participants:
            participants = list(set(memcell.participants))
        
        # ===== 第三步：根据场景决定是否提取个人 Episode =====
        if is_assistant_scene:
            logger.info(f"[Worker] 检测到 assistant 场景，仅提取群组 Episode，跳过个人 Episode")
            episode_tasks = [
                # 只提取群组 Episode (user_id=None)
                memory_manager.extract_memory(
                    memcell=memcell,
                    memory_type=MemoryType.EPISODIC_MEMORY,
                    user_id=None,
                    group_id=request.group_id,
                    group_name=request.group_name,
                ),
            ]
        else:
            logger.info(f"[Worker] 非 assistant 场景，并行提取群组 Episode 和 {len(participants)} 个个人 Episode")
            episode_tasks = [
                # 群组 Episode (user_id=None)
                memory_manager.extract_memory(
                    memcell=memcell,
                    memory_type=MemoryType.EPISODIC_MEMORY,
                    user_id=None,
                    group_id=request.group_id,
                    group_name=request.group_name,
                ),
                # 个人 Episodes
                *[
                    memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.EPISODIC_MEMORY,
                        user_id=user_id,
                        group_id=request.group_id,
                        group_name=request.group_name,
                    )
                    for user_id in participants
                ]
            ]
        
        episode_results = await asyncio.gather(*episode_tasks, return_exceptions=True)
        
        # ===== 第四步：处理 Episode 提取结果 =====
        group_episode_memories: List[Memory] = []
        group_episode = episode_results[0] if episode_results else None
        
        if isinstance(group_episode, Exception):
            logger.error(f"[Worker] ❌ 群组 Episode 提取异常: {group_episode}")
            group_episode = None
        elif group_episode:
            group_episode.ori_event_id_list = [memcell.event_id]
            group_episode.memcell_event_id_list = [memcell.event_id]
            group_episode_memories.append(group_episode)
            logger.info(f"[Worker] ✅ 群组 Episode 提取成功")
            
            memcell.episode = group_episode.episode
            memcell.subject = group_episode.subject
        else:
            logger.warning(f"[Worker] ⚠️  群组 Episode 提取失败")
        
        # 处理个人 Episodes
        episode_memories: List[Memory] = []
        if not is_assistant_scene:
            for user_id, result in zip(participants, episode_results[1:]):
                if isinstance(result, Exception):
                    logger.error(f"[Worker] ❌ 个人 Episode 提取异常: user_id={user_id}, error={result}")
                    continue
                
                if result:
                    result.ori_event_id_list = [memcell.event_id]
                    result.memcell_event_id_list = [memcell.event_id]
                    episode_memories.append(result)
                    logger.info(f"[Worker] ✅ 个人 Episode 提取成功: user_id={user_id}")
                else:
                    logger.warning(f"[Worker] ⚠️  个人 Episode 提取失败: user_id={user_id}")
        else:
            logger.info(f"[Worker] assistant 场景，跳过个人 Episode 处理")
        
        # ===== 第五步：更新 MemCell 的 episode 字段到数据库 =====
        if request.group_id and group_episode:
            try:
                memcell_repo = get_bean_by_type(MemCellRawRepository)
                updated_memcell = await memcell_repo.update_by_event_id(
                    event_id=memcell.event_id,
                    update_data={
                        "episode": group_episode.episode,
                        "subject": group_episode.subject,
                    }
                )
                if updated_memcell:
                    logger.info(f"[Worker] ✅ 更新 MemCell 的 episode 字段: {memcell.event_id}")
                else:
                    logger.warning(f"[Worker] ⚠️  未找到 MemCell: {memcell.event_id}")
            except Exception as e:
                logger.error(f"[Worker] ❌ 更新 MemCell episode 失败: {e}")
            
            # ===== 第六步：异步触发聚类 (不等待) =====
            try:
                memcell_for_clustering = MemCell(
                    event_id=memcell.event_id,
                    user_id_list=memcell.user_id_list,
                    original_data=memcell.original_data,
                    timestamp=memcell.timestamp,
                    summary=memcell.summary,
                    group_id=memcell.group_id,
                    group_name=memcell.group_name,
                    participants=memcell.participants,
                    type=memcell.type,
                    episode=group_episode.episode,
                )
                
                asyncio.create_task(_trigger_clustering(request.group_id, memcell_for_clustering, scene))
                logger.info(f"[Worker] 异步触发聚类任务 (scene={scene})")
            except Exception as e:
                logger.error(f"[Worker] ❌ 触发聚类失败: {e}")
        
        # ===== 第七步：保存 Episode 到数据库 =====
        if if_memorize(memcell):
            old_memory_list = await load_core_memories(request, participants, current_time)
            
            semantic_memories: List[SemanticMemoryItem] = []
            event_logs: List[EventLog] = []
            parent_docs_map: Dict[str, Any] = {}
            # episodic_source_memories 用于后续提取 semantic/eventlog
            episodic_source_memories: List[Memory] = group_episode_memories + episode_memories
            
            # 要保存的 Episode 列表（可能比 episodic_source_memories 多）
            episodes_to_save: List[Memory] = list(episodic_source_memories)
            
            # assistant 场景：额外复制群组 Episode 给每个用户（只保存，不参与提取）
            if is_assistant_scene and group_episode_memories:
                group_ep = group_episode_memories[0]
                for user_id in participants:
                    if "robot" in user_id.lower() or "assistant" in user_id.lower():
                        continue
                    user_ep = self._clone_memory_for_user(group_ep, user_id)
                    episodes_to_save.append(user_ep)
                logger.info(f"[Worker] assistant 场景，复制群组 Episode 给 {len(episodes_to_save) - len(episodic_source_memories)} 个用户")
            
            if episodes_to_save:
                for episode_mem in episodes_to_save:
                    if getattr(episode_mem, "group_name", None) is None:
                        episode_mem.group_name = request.group_name
                    if getattr(episode_mem, "user_name", None) is None:
                        episode_mem.user_name = episode_mem.user_id
                
                episodic_docs = [
                    _convert_episode_memory_to_doc(episode_mem, current_time)
                    for episode_mem in episodes_to_save
                ]
                episodic_payloads = [
                    MemoryDocPayload(MemoryType.EPISODIC_MEMORY, doc)
                    for doc in episodic_docs
                ]
                saved_docs_map = await save_memory_docs(episodic_payloads)
                saved_episode_docs = saved_docs_map.get(MemoryType.EPISODIC_MEMORY, [])
                
                # 只给 episodic_source_memories 设置 event_id（用于后续提取）
                for episode_mem, saved_doc in zip(episodic_source_memories, saved_episode_docs):
                    episode_mem.event_id = str(saved_doc.event_id)
                    parent_docs_map[str(saved_doc.event_id)] = saved_doc
            
            # ===== 第八步：并行提取 Semantic 和 EventLog =====
            logger.info(f"[Worker] 开始并行提取 Semantic 和 EventLog，共 {len(episodic_source_memories)} 个 Episode")
            
            extraction_tasks = []
            task_metadata = []
            
            for episode_mem in episodic_source_memories:
                if not episode_mem.event_id:
                    logger.warning(f"[Worker] ⚠️  Episode 缺少 event_id，跳过")
                    continue
                
                extraction_tasks.append(
                    memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.SEMANTIC_MEMORY,
                        user_id=episode_mem.user_id,
                        episode_memory=episode_mem,
                    )
                )
                task_metadata.append({
                    'type': MemoryType.SEMANTIC_MEMORY,
                    'episode_mem': episode_mem,
                })
                
                extraction_tasks.append(
                    memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.PERSONAL_EVENT_LOG,
                        user_id=episode_mem.user_id,
                        episode_memory=episode_mem,
                    )
                )
                task_metadata.append({
                    'type': MemoryType.PERSONAL_EVENT_LOG,
                    'episode_mem': episode_mem,
                })
            
            if extraction_tasks:
                extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
                
                for metadata, result in zip(task_metadata, extraction_results):
                    memory_type = metadata['type']
                    episode_mem = metadata['episode_mem']
                    
                    if isinstance(result, Exception):
                        logger.error(
                            f"[Worker] ❌ 提取异常: user_id={episode_mem.user_id}, "
                            f"memory_type={memory_type}, error={result}"
                        )
                        continue
                    
                    if not result:
                        logger.warning(
                            f"[Worker] ⚠️  提取失败或为空: user_id={episode_mem.user_id}, "
                            f"memory_type={memory_type}"
                        )
                        continue
                    
                    logger.info(
                        f"[Worker] ✅ 成功提取: user_id={episode_mem.user_id}, "
                        f"memory_type={memory_type}, "
                        f"数量={len(result) if isinstance(result, list) else 1}"
                    )
                    
                    if memory_type == MemoryType.SEMANTIC_MEMORY:
                        for mem in result:
                            mem.parent_event_id = episode_mem.event_id
                            mem.user_id = episode_mem.user_id
                            mem.group_id = episode_mem.group_id
                            mem.group_name = episode_mem.group_name
                            if getattr(mem, "user_name", None) is None:
                                mem.user_name = episode_mem.user_name
                            semantic_memories.append(mem)
                    elif memory_type == MemoryType.PERSONAL_EVENT_LOG:
                        result.parent_event_id = episode_mem.event_id
                        result.user_id = episode_mem.user_id
                        result.group_id = episode_mem.group_id
                        result.group_name = episode_mem.group_name
                        if getattr(result, "user_name", None) is None:
                            result.user_name = episode_mem.user_name
                        event_logs.append(result)
            
            # ===== 第九步：保存 Semantic 和 EventLog =====
            semantic_docs = []
            for sem_mem in semantic_memories:
                parent_doc = parent_docs_map.get(str(sem_mem.parent_event_id))
                if not parent_doc:
                    logger.warning(
                        f"[Worker] ⚠️  未找到 parent_event_id={sem_mem.parent_event_id} 对应的 episodic_memory"
                    )
                    continue
                doc = _convert_semantic_memory_to_doc(sem_mem, parent_doc, current_time)
                semantic_docs.append(doc)
            
            event_log_docs = []
            for event_log in event_logs:
                parent_doc = parent_docs_map.get(str(event_log.parent_event_id))
                if not parent_doc:
                    logger.warning(
                        f"[Worker] ⚠️  未找到 parent_event_id={event_log.parent_event_id} 对应的 episodic_memory"
                    )
                    continue
                docs = _convert_event_log_to_docs(event_log, parent_doc, current_time)
                event_log_docs.extend(docs)
            
            # assistant 场景：Semantic 和 EventLog 也复制给每个用户
            if is_assistant_scene:
                user_ids = [u for u in participants if "robot" not in u.lower() and "assistant" not in u.lower()]
                extra_semantic = [doc.model_copy(update={"user_id": uid, "user_name": uid}) for doc in semantic_docs for uid in user_ids]
                extra_eventlog = [doc.model_copy(update={"user_id": uid, "user_name": uid}) for doc in event_log_docs for uid in user_ids]
                semantic_docs.extend(extra_semantic)
                event_log_docs.extend(extra_eventlog)
                logger.info(f"[Worker] assistant 场景，复制 Semantic/EventLog 给 {len(user_ids)} 个用户")
            
            payloads: List[MemoryDocPayload] = []
            if semantic_docs:
                payloads.extend(
                    MemoryDocPayload(MemoryType.SEMANTIC_MEMORY, doc)
                    for doc in semantic_docs
                )
            if event_log_docs:
                payloads.extend(
                    MemoryDocPayload(MemoryType.PERSONAL_EVENT_LOG, doc)
                    for doc in event_log_docs
                )
            if payloads:
                await save_memory_docs(payloads)
            
            # ===== 第十步：更新状态表 =====
            await update_status_after_memcell(
                request, memcell, current_time, request.raw_data_type
            )
