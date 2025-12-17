#!/bin/bash
# scripts/deploy.sh

set -e

cd "$(dirname "$0")/.."

echo "🚀 Deploying DUM-E Vision Server"

# 체크포인트 확인
if [ ! -f "models/gdino/model.pth" ]; then
    echo "❌ GroundingDINO checkpoint not found!"
    echo "Please place model.pth in models/gdino/"
    exit 1
fi

if [ ! -f "models/gdino/config.py" ]; then
    echo "❌ GroundingDINO config not found!"
    echo "Please place config.py in models/gdino/"
    exit 1
fi

# Build and deploy
echo "📦 Building vision service..."
docker compose -f docker/compose.yaml build vision

echo "🚀 Starting vision service..."
docker compose -f docker/compose.yaml up -d vision

echo ""
echo "⏳ Waiting for service to be ready..."
sleep 10

# Health check
echo ""
echo "🏥 Checking service health..."
echo -n "Vision service: "
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Running"
    curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
else
    echo "❌ Failed"
    echo ""
    echo "📋 Recent logs:"
    docker compose -f docker/compose.yaml logs --tail=30 vision
fi

echo ""
echo "📊 Container status:"
docker compose -f docker/compose.yaml ps

echo ""
echo "📝 Useful commands:"
echo "  View logs:     docker compose -f docker/compose.yaml logs -f vision"
echo "  Stop service:  docker compose -f docker/compose.yaml down"
echo "  Restart:       docker compose -f docker/compose.yaml restart vision"
echo "  Rebuild:       docker compose -f docker/compose.yaml build vision"
echo ""
echo "✅ Deployment complete!"
