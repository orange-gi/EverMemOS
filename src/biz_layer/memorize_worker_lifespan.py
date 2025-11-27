"""
记忆提取 Worker 生命周期管理
"""

from core.lifespan.lifespan_interface import LifespanProvider
from core.observation.logger import get_logger
from core.di.decorators import component

logger = get_logger(__name__)


@component(name="memorize_worker_lifespan_provider")
class MemorizeWorkerLifespanProvider(LifespanProvider):
    
    def __init__(self, name: str = "memorize_worker", order: int = 100):
        super().__init__(name, order)
    
    async def startup(self, app):
        from core.di import get_bean_by_type
        from biz_layer.memorize_worker_service import MemorizeWorkerService
        worker = get_bean_by_type(MemorizeWorkerService)
        await worker.start()
        logger.info("[MemorizeWorker] ✅ 已启动")
    
    async def shutdown(self, app):
        from core.di import get_bean_by_type
        from biz_layer.memorize_worker_service import MemorizeWorkerService
        worker = get_bean_by_type(MemorizeWorkerService)
        await worker.stop()
        logger.info("[MemorizeWorker] ✅ 已停止")

