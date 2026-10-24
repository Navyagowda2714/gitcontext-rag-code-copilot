# Deployment

## Environments

We deploy to `staging` on every merge to `main` and promote to `production` manually after the
smoke tests pass. Production runs 12 API pods behind the load balancer.

## Configuration

All configuration comes from environment variables prefixed with `SHOPFRONT_` (see
`shopfront/config.py`). `Settings.validate()` runs at start-up and refuses to boot with unsafe
rate-limit values; a pod that fails validation never becomes ready, so a bad config cannot roll
out past the first canary.

## Rollback

Use `make rollback ENV=production` to redeploy the previous image tag. Database migrations must be
backwards compatible for one release so that rollbacks are always safe.
