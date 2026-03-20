import redis.asyncio as aioredis
from app.core.config import settings

redis_client: aioredis.Redis = None

async def get_redis() -> aioredis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    return redis_client

async def close_redis():
    global redis_client
    if redis_client:
        await redis_client.aclose()
        redis_client = None