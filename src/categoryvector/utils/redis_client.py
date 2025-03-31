import redis
from typing import Any, Optional, Union, List
import json
from ..config import CategoryVectorConfig

class RedisClient:
    """Redis 客户端工具类"""
    
    def __init__(self):
        """初始化 Redis 客户端"""
        config = CategoryVectorConfig.get_redis_config()
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6379)
        self.db = config.get('db', 0)
        self.prefix = config.get('prefix', '')
        self.ttl = config.get('ttl', 0)
        
        try:
            print(f"正在连接 Redis: {self.host}:{self.port} (数据库: {self.db})...")
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=config.get('password', ''),
                socket_timeout=config.get('socket_timeout', 5),
                socket_connect_timeout=config.get('socket_connect_timeout', 5),
                retry_on_timeout=config.get('retry_on_timeout', True),
                decode_responses=True  # 自动将字节解码为字符串
            )
            
            # 测试连接
            self.client.ping()
            print(f"✓ Redis 连接成功")
            if self.prefix:
                print(f"  - 键前缀: {self.prefix}")
            if self.ttl > 0:
                print(f"  - 过期时间: {self.ttl}秒")
            
        except Exception as e:
            print(f"✗ Redis 连接失败: {e}")
            # 仍然创建实例，但部分功能可能无法使用
            self.client = None
    
    def _get_key(self, key: str) -> str:
        """获取带前缀的完整键名"""
        return f"{self.prefix}{key}"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置键值对
        
        Args:
            key: 键名
            value: 值
            ttl: 过期时间（秒），默认使用配置文件中的 ttl
            
        Returns:
            bool: 是否设置成功
        """
        if self.client is None:
            return False
            
        try:
            key = self._get_key(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
                
            # 使用传入的 ttl 或默认的实例 ttl
            expiration = ttl if ttl is not None else self.ttl
            
            return self.client.set(key, value, ex=expiration if expiration > 0 else None)
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取键值
        
        Args:
            key: 键名
            
        Returns:
            Any: 值，如果键不存在则返回 None
        """
        try:
            key = self._get_key(key)
            value = self.client.get(key)
            if value is None:
                return None
            try:
                return json.loads(value)
            except:
                return value
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        删除键
        
        Args:
            key: 键名
            
        Returns:
            bool: 是否删除成功
        """
        try:
            key = self._get_key(key)
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 键名
            
        Returns:
            bool: 键是否存在
        """
        try:
            key = self._get_key(key)
            return bool(self.client.exists(key))
        except Exception as e:
            print(f"Redis exists error: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        设置键的过期时间
        
        Args:
            key: 键名
            ttl: 过期时间（秒）
            
        Returns:
            bool: 是否设置成功
        """
        try:
            key = self._get_key(key)
            return bool(self.client.expire(key, ttl))
        except Exception as e:
            print(f"Redis expire error: {e}")
            return False
    
    def ttl(self, key: str) -> Optional[int]:
        """
        获取键的剩余过期时间
        
        Args:
            key: 键名
            
        Returns:
            Optional[int]: 剩余过期时间（秒），如果键不存在则返回 None
        """
        try:
            key = self._get_key(key)
            ttl = self.client.ttl(key)
            return ttl if ttl >= 0 else None
        except Exception as e:
            print(f"Redis ttl error: {e}")
            return None
    
    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        增加键的值
        
        Args:
            key: 键名
            amount: 增加的数量
            
        Returns:
            Optional[int]: 增加后的值，如果键不存在则返回 None
        """
        try:
            key = self._get_key(key)
            return self.client.incr(key, amount)
        except Exception as e:
            print(f"Redis incr error: {e}")
            return None
    
    def decr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        减少键的值
        
        Args:
            key: 键名
            amount: 减少的数量
            
        Returns:
            Optional[int]: 减少后的值，如果键不存在则返回 None
        """
        try:
            key = self._get_key(key)
            return self.client.decr(key, amount)
        except Exception as e:
            print(f"Redis decr error: {e}")
            return None
    
    def hset(self, key: str, field: str, value: Any) -> bool:
        """
        设置哈希字段
        
        Args:
            key: 键名
            field: 字段名
            value: 值
            
        Returns:
            bool: 是否设置成功
        """
        try:
            key = self._get_key(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return bool(self.client.hset(key, field, value))
        except Exception as e:
            print(f"Redis hset error: {e}")
            return False
    
    def hget(self, key: str, field: str) -> Optional[Any]:
        """
        获取哈希字段值
        
        Args:
            key: 键名
            field: 字段名
            
        Returns:
            Any: 字段值，如果字段不存在则返回 None
        """
        try:
            key = self._get_key(key)
            value = self.client.hget(key, field)
            if value is None:
                return None
            try:
                return json.loads(value)
            except:
                return value
        except Exception as e:
            print(f"Redis hget error: {e}")
            return None
    
    def hdel(self, key: str, field: str) -> bool:
        """
        删除哈希字段
        
        Args:
            key: 键名
            field: 字段名
            
        Returns:
            bool: 是否删除成功
        """
        try:
            key = self._get_key(key)
            return bool(self.client.hdel(key, field))
        except Exception as e:
            print(f"Redis hdel error: {e}")
            return False
    
    def hgetall(self, key: str) -> dict:
        """
        获取哈希所有字段和值
        
        Args:
            key: 键名
            
        Returns:
            dict: 字段和值的字典
        """
        try:
            key = self._get_key(key)
            result = self.client.hgetall(key)
            return {k: json.loads(v) if v.startswith('{') or v.startswith('[') else v 
                   for k, v in result.items()}
        except Exception as e:
            print(f"Redis hgetall error: {e}")
            return {}
    
    def sadd(self, key: str, *values: Any) -> bool:
        """
        向集合添加元素
        
        Args:
            key: 键名
            *values: 要添加的值
            
        Returns:
            bool: 是否添加成功
        """
        try:
            key = self._get_key(key)
            values = [json.dumps(v) if isinstance(v, (dict, list)) else str(v) for v in values]
            return bool(self.client.sadd(key, *values))
        except Exception as e:
            print(f"Redis sadd error: {e}")
            return False
    
    def srem(self, key: str, *values: Any) -> bool:
        """
        从集合中移除元素
        
        Args:
            key: 键名
            *values: 要移除的值
            
        Returns:
            bool: 是否移除成功
        """
        try:
            key = self._get_key(key)
            values = [json.dumps(v) if isinstance(v, (dict, list)) else str(v) for v in values]
            return bool(self.client.srem(key, *values))
        except Exception as e:
            print(f"Redis srem error: {e}")
            return False
    
    def smembers(self, key: str) -> set:
        """
        获取集合所有成员
        
        Args:
            key: 键名
            
        Returns:
            set: 集合成员
        """
        try:
            key = self._get_key(key)
            members = self.client.smembers(key)
            return {json.loads(m) if m.startswith('{') or m.startswith('[') else m 
                   for m in members}
        except Exception as e:
            print(f"Redis smembers error: {e}")
            return set()
    
    def sismember(self, key: str, value: Any) -> bool:
        """
        检查值是否是集合成员
        
        Args:
            key: 键名
            value: 要检查的值
            
        Returns:
            bool: 是否是集合成员
        """
        try:
            key = self._get_key(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return bool(self.client.sismember(key, value))
        except Exception as e:
            print(f"Redis sismember error: {e}")
            return False
    
    def close(self):
        """关闭 Redis 连接"""
        try:
            self.client.close()
        except Exception as e:
            print(f"Redis close error: {e}")

# 创建全局 Redis 客户端实例
redis_client = RedisClient() 