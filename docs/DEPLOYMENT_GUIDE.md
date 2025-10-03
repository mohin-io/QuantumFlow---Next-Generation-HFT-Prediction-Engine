# QuantumFlow - Production Deployment Guide

### Complete guide for deploying the QuantumFlow HFT Prediction Engine to production

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [AWS Deployment](#aws-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring & Alerting](#monitoring--alerting)
6. [Security Hardening](#security-hardening)
7. [Disaster Recovery](#disaster-recovery)

---

## Pre-Deployment Checklist

### Code Preparation

- [ ] All unit tests passing (`pytest tests/`)
- [ ] Integration tests passing (`pytest tests/test_integration.py`)
- [ ] Code coverage >80%
- [ ] No secrets in code (use environment variables)
- [ ] Dependencies pinned in requirements.txt
- [ ] Docker images built and tested
- [ ] Database migrations ready

### Infrastructure

- [ ] Cloud account configured (AWS/GCP)
- [ ] Domain name registered
- [ ] SSL certificates obtained
- [ ] Monitoring tools setup (CloudWatch/Stackdriver)
- [ ] Backup strategy defined
- [ ] Disaster recovery plan documented

### Performance

- [ ] Load testing completed
- [ ] Latency targets met (<50ms p99)
- [ ] Throughput validated (>1000 req/s)
- [ ] Database indexes created
- [ ] Caching implemented (Redis)

---

## AWS Deployment

### Option 1: CloudFormation (Recommended)

**Deploy complete stack:**

```bash
# Navigate to deployment directory
cd deploy/aws

# Validate template
aws cloudformation validate-template \
  --template-body file://cloudformation-stack.yaml

# Create stack
aws cloudformation create-stack \
  --stack-name hft-production \
  --template-body file://cloudformation-stack.yaml \
  --parameters \
    ParameterKey=EnvironmentName,ParameterValue=production \
    ParameterKey=InstanceType,ParameterValue=t3.xlarge \
    ParameterKey=KeyPairName,ParameterValue=your-keypair \
  --capabilities CAPABILITY_IAM

# Monitor deployment
aws cloudformation describe-stacks \
  --stack-name hft-production \
  --query 'Stacks[0].StackStatus'

# Get outputs
aws cloudformation describe-stacks \
  --stack-name hft-production \
  --query 'Stacks[0].Outputs'
```

**Resources Created:**
- VPC with public/private subnets
- Application Load Balancer
- Auto Scaling Group (2-10 instances)
- RDS TimescaleDB (db.t3.medium)
- ElastiCache Redis cluster
- S3 bucket for data
- IAM roles and security groups

**Estimated Cost:** $500-1500/month (depending on traffic)

### Option 2: ECS Fargate

```bash
# Create ECR repositories
aws ecr create-repository --repository-name hft-api
aws ecr create-repository --repository-name hft-dashboard
aws ecr create-repository --repository-name hft-ingestion

# Build and push images
docker build -f docker/Dockerfile.api -t hft-api .
docker tag hft-api:latest $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/hft-api:latest
docker push $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/hft-api:latest

# Create ECS cluster
aws ecs create-cluster --cluster-name hft-cluster

# Deploy services (use ECS console or CLI)
```

---

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Deploy to AWS EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name hft-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --name hft-cluster --region us-east-1

# Deploy application
kubectl apply -f deploy/kubernetes/deployment.yaml

# Check status
kubectl get pods -n hft-production
kubectl get services -n hft-production

# Get LoadBalancer URL
kubectl get service hft-api -n hft-production \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

### Deploy to GCP GKE

```bash
# Create GKE cluster
gcloud container clusters create hft-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials hft-cluster --zone us-central1-a

# Deploy application
kubectl apply -f deploy/kubernetes/deployment.yaml
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment hft-api --replicas=10 -n hft-production

# Horizontal Pod Autoscaling (already configured in deployment.yaml)
kubectl get hpa -n hft-production

# Cluster autoscaling (configure node pool)
kubectl autoscale deployment hft-api \
  --cpu-percent=70 \
  --min=3 \
  --max=20 \
  -n hft-production
```

---

## Performance Optimization

### Database Optimization

**Create indexes:**

```python
# Run optimization script
python scripts/optimize_performance.py

# Or manually execute SQL
psql -h your-db-host -U hftadmin -d hft_orderbook < deploy/sql/create_indexes.sql
```

**TimescaleDB tuning:**

```sql
-- Increase shared buffers (25% of RAM)
ALTER SYSTEM SET shared_buffers = '8GB';

-- Increase work memory
ALTER SYSTEM SET work_mem = '256MB';

-- Enable parallel queries
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Restart PostgreSQL
pg_ctl restart
```

### Application Optimization

**Enable Redis caching:**

```python
# In your application
from caching import CachingStrategy
import redis

redis_client = redis.Redis(host='redis', port=6379)
cache = CachingStrategy(redis_client)

# Cache predictions
cache.cache_prediction(symbol, timestamp, prediction, ttl=60)
```

**Use Numba for features:**

```python
from optimize_performance import OptimizedFeatureCalculator

calculator = OptimizedFeatureCalculator()
features = calculator.calculate_features_batch(order_books)
```

**Enable gzip compression:**

```python
# In FastAPI app
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### Infrastructure Optimization

**CloudFront CDN (AWS):**

```bash
# Create CloudFront distribution
aws cloudfront create-distribution \
  --origin-domain-name your-alb-dns.amazonaws.com \
  --default-root-object index.html
```

**Load Balancer tuning:**

- Enable connection draining (300s)
- Increase idle timeout (60s)
- Enable cross-zone load balancing
- Use target groups with health checks

---

## Monitoring & Alerting

### Prometheus + Grafana

**Deploy monitoring stack:**

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Default credentials: admin/prom-operator
```

**Custom metrics:**

```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@prediction_latency.time()
def make_prediction(features):
    prediction_counter.inc()
    # ... prediction logic
```

### CloudWatch (AWS)

**Custom metrics:**

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='HFT',
    MetricData=[
        {
            'MetricName': 'PredictionLatency',
            'Value': latency_ms,
            'Unit': 'Milliseconds'
        }
    ]
)
```

**Alarms:**

```bash
# Create latency alarm
aws cloudwatch put-metric-alarm \
  --alarm-name high-latency \
  --alarm-description "Alert when p99 latency > 100ms" \
  --metric-name TargetResponseTime \
  --namespace AWS/ApplicationELB \
  --statistic Average \
  --period 300 \
  --threshold 0.1 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

### Logging

**Centralized logging with ELK:**

```yaml
# Deploy Elasticsearch + Kibana
kubectl apply -f https://download.elastic.co/downloads/eck/2.9.0/crds.yaml
kubectl apply -f https://download.elastic.co/downloads/eck/2.9.0/operator.yaml
```

**Application logging:**

```python
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

logger.info('Prediction made', extra={
    'symbol': 'BTCUSDT',
    'latency_ms': 25.5,
    'confidence': 0.87
})
```

---

## Security Hardening

### Network Security

**Security groups (AWS):**

```bash
# Allow only HTTPS
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# Database access only from app tier
aws ec2 authorize-security-group-ingress \
  --group-id sg-db \
  --protocol tcp \
  --port 5432 \
  --source-group sg-app
```

**Network policies (Kubernetes):**

Already configured in `deploy/kubernetes/deployment.yaml`

### Secrets Management

**AWS Secrets Manager:**

```bash
# Store database password
aws secretsmanager create-secret \
  --name hft/db/password \
  --secret-string "your-secure-password"

# Retrieve in application
```

```python
import boto3

client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='hft/db/password')
db_password = response['SecretString']
```

**Kubernetes Secrets:**

```bash
# Create secret
kubectl create secret generic hft-secrets \
  --from-literal=db-password=your-password \
  -n hft-production

# Use in pod (already configured in deployment.yaml)
```

### SSL/TLS

**Let's Encrypt (free):**

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f deploy/kubernetes/cluster-issuer.yaml

# Certificates auto-renew
```

**AWS Certificate Manager:**

```bash
# Request certificate
aws acm request-certificate \
  --domain-name api.yourdomain.com \
  --validation-method DNS
```

### API Security

**Rate limiting:**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/predict")
@limiter.limit("100/minute")
async def predict():
    # ... prediction logic
```

**API authentication:**

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_SECRET_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
```

---

## Disaster Recovery

### Backup Strategy

**Database backups:**

```bash
# Automated RDS backups (AWS)
aws rds modify-db-instance \
  --db-instance-identifier hft-db \
  --backup-retention-period 30 \
  --preferred-backup-window "03:00-04:00"

# Manual snapshot
aws rds create-db-snapshot \
  --db-instance-identifier hft-db \
  --db-snapshot-identifier hft-db-snapshot-$(date +%Y%m%d)
```

**PostgreSQL manual backup:**

```bash
# Dump database
pg_dump -h db-host -U hftadmin hft_orderbook > backup.sql

# Upload to S3
aws s3 cp backup.sql s3://hft-backups/$(date +%Y%m%d)/
```

### Recovery Procedures

**Database restore:**

```bash
# From RDS snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier hft-db-restored \
  --db-snapshot-identifier hft-db-snapshot-20250101

# From SQL dump
psql -h new-db-host -U hftadmin hft_orderbook < backup.sql
```

**Application rollback:**

```bash
# Kubernetes rollback
kubectl rollout undo deployment/hft-api -n hft-production

# ECS rollback
aws ecs update-service \
  --cluster hft-cluster \
  --service hft-api \
  --task-definition hft-api:previous-version
```

### Multi-Region Deployment

**AWS:**

- Primary: us-east-1
- Secondary: us-west-2
- Use Route53 for failover
- Enable cross-region RDS replication

**GCP:**

- Primary: us-central1
- Secondary: us-east1
- Use Cloud Load Balancing
- Enable cross-region SQL replication

---

## Post-Deployment

### Health Checks

```bash
# Check API
curl https://api.yourdomain.com/health

# Check database
psql -h db-host -U hftadmin -c "SELECT 1"

# Check Redis
redis-cli -h redis-host ping
```

### Load Testing

```bash
# Install artillery
npm install -g artillery

# Run load test
artillery quick --count 100 --num 1000 https://api.yourdomain.com/predict
```

### Performance Validation

```python
# Run benchmarks
python scripts/optimize_performance.py

# Expected results:
# - Feature extraction: >500 snapshots/sec
# - Model inference: <50ms p99
# - API response: <100ms p99
```

---

## Cost Optimization

### AWS Cost Estimate

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| EC2 (t3.xlarge x3) | On-Demand | $450 |
| RDS (db.t3.medium) | Multi-AZ | $280 |
| ElastiCache (cache.t3.medium) | 2 nodes | $100 |
| ALB | Standard | $25 |
| Data Transfer | 1TB/month | $90 |
| S3 Storage | 100GB | $3 |
| **Total** | | **~$950/month** |

**Savings:**
- Use Reserved Instances: -40% ($570/month)
- Use Spot Instances for workers: -70% ($135/month)
- Optimize data transfer: -50% ($45/month saved)

### GCP Cost Estimate

Similar costs, with sustained use discounts automatically applied.

---

## Support & Troubleshooting

**Common Issues:**

1. **High latency**: Check database indexes, enable caching
2. **Memory errors**: Increase instance size or optimize code
3. **Database connection errors**: Check security groups, increase connection pool
4. **503 errors**: Scale up instances, check health checks

**Monitoring:**

- Dashboard: Grafana/CloudWatch
- Alerts: PagerDuty/Slack
- Logs: CloudWatch Logs/ELK Stack

**Contact:**

- GitHub Issues: https://github.com/mohin-io/hft-order-book-imbalance/issues
- Documentation: See docs/ folder

---

**Version**: 1.0.0
**Last Updated**: October 2025
**License**: MIT
