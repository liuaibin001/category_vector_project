# CategoryVector 部署指南

## 1. 环境要求

### 1.1 硬件要求
- CPU: 4核或以上
- 内存: 16GB或以上
- 硬盘: 50GB或以上（取决于数据量）
- 网络: 100Mbps以上

### 1.2 软件要求
- Ubuntu 20.04 LTS或更高版本
- Python 3.8或更高版本
- Docker 20.10或更高版本
- Docker Compose 2.0或更高版本

## 2. 安装基础环境

### 2.1 安装Python环境
```bash
# 安装Python和相关工具
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3-pip

# 安装Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

### 2.2 安装Docker和Docker Compose
```bash
# 安装Docker
curl -fsSL https://get.docker.com | sh

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## 3. 部署Milvus

### 3.1 创建docker-compose.yml
```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

### 3.2 启动Milvus
```bash
# 创建目录
mkdir -p milvus/volumes
cd milvus

# 启动服务
docker-compose up -d
```

## 4. 部署Redis

### 4.1 使用Docker启动Redis
```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  -v redis-data:/data \
  redis:7.0 \
  redis-server --appendonly yes
```

## 5. 部署CategoryVector

### 5.1 准备项目
```bash
# 创建项目目录
mkdir -p /opt/categoryvector
cd /opt/categoryvector

# 克隆项目（如果使用Git）
git clone <your-repository-url> .

# 或者直接复制项目文件
scp -r /path/to/your/project/* user@server:/opt/categoryvector/
```

### 5.2 创建配置文件
```bash
# 创建配置文件
cat > config.toml << EOF
[milvus]
host = "localhost"
port = "19530"
collection_name = "category_vectors"

[search]
threshold = 0.3
top_k = 10
similarity_metric = "IP"

[index]
type = "flat"
nlist = 100
m_factor = 16

[model]
name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
vector_dim = 384

[logging]
level = "INFO"
file = "logs/categoryvector.log"

[data]
output_dir = "data/vectors"
EOF
```

### 5.3 安装依赖
```bash
# 创建虚拟环境并安装依赖
cd /opt/categoryvector
poetry install --no-dev
```

### 5.4 创建systemd服务
```bash
# 创建服务文件
sudo cat > /etc/systemd/system/categoryvector.service << EOF
[Unit]
Description=CategoryVector API Service
After=network.target

[Service]
Type=simple
User=categoryvector
Group=categoryvector
WorkingDirectory=/opt/categoryvector
Environment=PATH=/opt/categoryvector/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/categoryvector/.venv/bin/poetry run serve --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=1
StartLimitInterval=0

[Install]
WantedBy=multi-user.target
EOF

# 创建用户和组
sudo useradd -r -s /bin/false categoryvector
sudo chown -R categoryvector:categoryvector /opt/categoryvector

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable categoryvector
sudo systemctl start categoryvector
```

### 5.5 配置Nginx（可选）
```bash
# 安装Nginx
sudo apt install -y nginx

# 创建Nginx配置
sudo cat > /etc/nginx/sites-available/categoryvector << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# 启用配置
sudo ln -s /etc/nginx/sites-available/categoryvector /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 6. 初始化数据

### 6.1 构建向量索引
```bash
cd /opt/categoryvector
poetry run cv build --categories data/valid_sample.json
```

## 7. 监控和维护

### 7.1 日志查看
```bash
# 查看服务日志
sudo journalctl -u categoryvector -f

# 查看应用日志
tail -f /opt/categoryvector/logs/categoryvector.log
```

### 7.2 服务管理
```bash
# 重启服务
sudo systemctl restart categoryvector

# 查看服务状态
sudo systemctl status categoryvector
```

### 7.3 数据备份
```bash
# 备份Milvus数据
cd /path/to/milvus
docker-compose stop
tar -czf milvus-backup-$(date +%Y%m%d).tar.gz volumes/
docker-compose start

# 备份Redis数据
docker exec redis redis-cli SAVE
```

## 8. 安全建议

1. 配置防火墙，只开放必要端口
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

2. 设置Redis密码
```bash
# 修改Redis配置
docker exec -it redis redis-cli
CONFIG SET requirepass your-strong-password
CONFIG REWRITE
```

3. 配置SSL证书（推荐使用Let's Encrypt）
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 9. 性能优化

1. 调整系统参数
```bash
# 增加文件描述符限制
sudo cat >> /etc/security/limits.conf << EOF
categoryvector soft nofile 65535
categoryvector hard nofile 65535
EOF
```

2. 优化Nginx配置
```bash
# 修改worker进程数
worker_processes auto;

# 调整worker连接数
events {
    worker_connections 2048;
}
```

3. 调整Python应用参数
```bash
# 使用多个worker
poetry run serve --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

## 10. 故障排除

1. 服务无法启动
- 检查日志: `journalctl -u categoryvector -n 100`
- 检查配置文件权限
- 验证依赖是否安装完整

2. Milvus连接问题
- 检查Milvus容器状态: `docker ps`
- 查看Milvus日志: `docker logs milvus-standalone`
- 验证网络连接: `telnet localhost 19530`

3. Redis连接问题
- 检查Redis容器状态: `docker ps`
- 查看Redis日志: `docker logs redis`
- 验证网络连接: `telnet localhost 6379` 