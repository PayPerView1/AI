---
category: analytics
role: public
source: views.md
---

# View Tracking & Verification

For Pay-per-View (PPV) campaigns, earnings are calculated based on verified views. The platform uses advanced systems to measure performance.

## How Views are Tracked
- **API Integration:** We connect directly to social network APIs (TikTok Graph API, YouTube Reporting API, etc.) to fetch view counts.
- **Polling Intervals:** View data is updated every 6 hours for the first 7 days, and daily for the subsequent 23 days.
- **Tracking Window:** Earnings accumulate during the first 30 days after the video link is submitted. Views after day 30 are not eligible for payouts.

## Anti-Fraud & Verification
- **Bot Detection:** Algorithms analyze historical view trajectory, geo-distributions, and engagement rates to filter out bot traffic.
- **Engagement Consistency:** A video with high views but zero likes, shares, or comments is flagged for manual audit.
- **Invalid Views:** Any views identified as fraudulent, purchased, or bot-generated will be deducted from the payout calculation. Repeated offences lead to account suspension.
