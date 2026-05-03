# Ads Engine API

Production-grade FastAPI backend with **15 AI-powered advertising endpoints** for complete ad campaign automation.

## What This Is

A comprehensive REST API powered by Claude AI that generates production-ready advertising strategies across all major platforms (Meta, Google Ads, TikTok, Apple Search Ads, YouTube). Each endpoint handles a critical stage of campaign setup, from audience segmentation to performance reporting.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Copy `.env.example` to `.env` and fill in your actual keys:

```bash
cp .env.example .env
```

Then edit `.env`:
- `ANTHROPIC_API_KEY`: Your Claude API key from [Anthropic](https://console.anthropic.com)
- `ADS_ENGINE_API_KEY`: Any secret key you choose for API authentication

### 3. Run the Server

```bash
uvicorn main:app --reload --port 8000
```

Server will be available at `http://localhost:8000`

## All 15 Endpoints

### Stage 1: Audience & Keywords (Foundational Strategy)
1. **POST /api/v1/ads-audience** - Define and segment target audiences across platforms
2. **POST /api/v1/ads-keywords** - Build comprehensive keyword strategy with intent classification

### Stage 2: Creative & Content (Campaign Assets)
3. **POST /api/v1/ads-copy** - Generate platform-optimized ad copy with psychological triggers
4. **POST /api/v1/ads-hooks** - Create scroll-stopping hooks for social and static ads
5. **POST /api/v1/ads-creative** - Generate visual creative asset specifications

### Stage 3: Video & Conversion (Extended Assets)
6. **POST /api/v1/ads-video** - Generate platform-optimized video ad scripts with scene breakdowns
7. **POST /api/v1/ads-landing** - Optimize landing pages for ad traffic conversion

### Stage 4: Funnel & Budget (Campaign Architecture)
8. **POST /api/v1/ads-funnel** - Map post-click customer journey across TOFU/MOFU/BOFU/Post-Install stages
9. **POST /api/v1/ads-budget** - Allocate budget across platforms, funnel stages, and campaigns

### Stage 5: Competitive Intelligence (Market Analysis)
10. **POST /api/v1/ads-competitors** - Analyze competitor ad strategies and identify positioning gaps

### Stage 6: Testing & Optimization (Experimentation)
11. **POST /api/v1/ads-testing** - Generate systematic A/B testing framework with statistical rigor
12. **POST /api/v1/ads-audit** - Generate campaign audit checklists (pre-launch, weekly, monthly)

### Stage 7: Planning & Reporting (Execution & Analytics)
13. **POST /api/v1/ads-strategy** - Generate comprehensive 90-day master strategy with phased approach
14. **POST /api/v1/ads-report** - Generate weekly/monthly performance reports with recommendations

### Stage 8: Quick Launch (Rapid Deployment)
15. **POST /api/v1/ads-quick** - Generate complete 1-hour campaign ready for immediate deployment

## Example: Quick Campaign Launch

```bash
curl -X POST http://localhost:8000/api/v1/ads-quick \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "product_description": "FastAI Health Coach - AI-powered intermittent fasting app",
    "daily_budget": 50,
    "platform": "meta",
    "launch_urgency": "immediate",
    "target_market": {
      "age_range": "25-55",
      "interests": ["health", "fitness", "nutrition"],
      "geography": ["US", "UK", "CA"]
    },
    "include_setup_checklist": true
  }'
```

## API Key Authentication

All endpoints require the `X-API-Key` header:

```bash
X-API-Key: your-ads-engine-api-key
```

## Documentation

- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **Root Info**: GET http://localhost:8000/

## Architecture

- **Framework**: FastAPI (async/concurrent requests)
- **LLM**: Anthropic Claude API (vision-capable, 200K context)
- **Auth**: API key via header (production: add JWT/OAuth)
- **Models**: Pydantic v2 with comprehensive validation
- **Async**: Full async/await pipeline for performance

## Environment Variables

```
ANTHROPIC_API_KEY=sk-ant-...              # Your Anthropic API key
ADS_ENGINE_API_KEY=your-secret-key        # Your API authentication key
```

## Development

### Run Tests
```bash
pytest tests/ -v
```

### Format Code
```bash
black .
```

### Type Checking
```bash
mypy main.py
```

## Production Deployment

For production use, add:
- Environment-based configuration (dev/staging/prod)
- Request logging and monitoring
- Rate limiting per API key
- Database persistence for request history
- Async worker queue (Celery) for long-running tasks
- Cloud storage for generated reports (S3, GCS)
- Authentication upgrades (JWT tokens with refresh)
- API versioning strategy

## Support

For issues or questions, refer to:
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
