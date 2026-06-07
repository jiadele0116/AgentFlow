# 部署指南

## 前置条件

- [Docker](https://docs.docker.com/get-docker/) >= 20.10
- [Docker Compose](https://docs.docker.com/compose/install/) >= 2.0
- 有效的 LLM API Key（如通义千问 DashScope）

## 快速启动

### 1. 配置环境变量

复制环境变量模板并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
OPENAI_API_KEY=your-actual-api-key
```

### 2. 构建并启动

```bash
# 构建镜像并启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 3. 访问服务

- **聊天界面**: http://localhost:8880
- **技术文档**: http://localhost:8880/guide
- **健康检查**: http://localhost:8880/health

## 常用命令

```bash
# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 重新构建（代码更新后）
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 进入容器调试
docker exec -it multi-agent-system bash
```

## 配置说明

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ | LLM API Key，支持 OpenAI 兼容接口 |

### 数据持久化

以下数据通过 Docker Volume 持久化，`docker-compose down` 不会丢失：

| 目录 | 说明 |
|------|------|
| `data/chroma/` | ChromaDB 向量数据库 |
| `data/workspace/` | Agent 工作目录 |

如需完全清除数据：

```bash
docker-compose down -v  # 同时删除 volumes
```

### 修改配置

编辑 `config.yaml` 后重启容器即可生效：

```bash
docker-compose restart
```

## 自定义部署

### 修改端口

编辑 `docker-compose.yml`，修改端口映射：

```yaml
ports:
  - "自定义端口:8880"
```

### 使用外部数据库

编辑 `config.yaml`，将 MCP 数据库配置启用并设置连接字符串。也可通过 `.env` 文件设置 `DB_URL` 环境变量。

## 故障排查

### 服务无法启动

```bash
# 查看详细日志
docker-compose logs -f app
```

### 常见问题

**1. 端口被占用**
```
Error: bind: address already in use
```
修改 `docker-compose.yml` 中的端口映射，或停止占用 8880 端口的进程。

**2. API Key 无效**
```
AuthenticationError
```
检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确。

**3. 镜像构建失败**
```bash
# 清除缓存重新构建
docker-compose build --no-cache
```

**4. 磁盘空间不足**
```bash
# 清理未使用的 Docker 资源
docker system prune
```

## 生产环境建议

1. **反向代理**: 使用 Nginx/Caddy 做反向代理，配置 HTTPS
2. **日志收集**: 将日志输出到文件或接入日志系统（ELK/Loki）
3. **监控**: 对 `/health` 端点配置监控告警
4. **备份**: 定期备份 `data/chroma/` 目录
5. **资源限制**: 在 `docker-compose.yml` 中添加 `deploy.resources.limits`
