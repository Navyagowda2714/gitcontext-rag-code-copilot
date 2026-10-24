---
id: INC-2026-022
title: Customers received duplicate invoices after webhook retries
severity: SEV-2
date: 2026-05-02
authors: [marcus-chen]
services: [shopfront-api, billing]
related_prs: [152]
---

# Postmortem: duplicate invoices

## Summary

1,340 customers received two invoices for the May billing period. No one was double-charged
because payment collection deduplicates on the subscription, but support received over 400
tickets.

## Root cause

A slow database migration made the webhook handler take about 14 seconds per `invoice.upcoming`
event. The payment provider retries any delivery that is not acknowledged within 10 seconds, so
most events were delivered twice. The handler was not idempotent: each delivery created an
invoice.

## Resolution

Voided the duplicate invoices with a script and emailed affected customers.

## Action items

- Deduplicate webhook deliveries by event id before side effects (PR #152, done).
- Acknowledge webhooks immediately and process them in a background job (BILL-311, in progress).
- Add an alert on webhook handler latency above 5 seconds (done).
