# EstateKaki — Town Council Telegram Bot (Semantic Hybrid RAG + Ticketing + Trello)

## Setup
1. `python -m venv .venv` and activate it (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in `TELEGRAM_TOKEN`, `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY`), and `STAFF_CHAT_ID` (send `/chatid` in your staff group).
4. *(Optional)* Set `TRELLO_BOARD_ID`, `TRELLO_API_KEY`, and `TRELLO_TOKEN` to auto-submit work orders to Trello (`To Do` / `Urgent`, plus optional `In Progress` / `Done` status syncing).
5. Build the semantic RAG index: `python ingest.py`
6. Start the bot: `python bot.py`
7. Start the staff operations dashboard in another terminal: `streamlit run dashboard.py`

## Try It
- *"Can I throw away my old sofa for free?"* -> Semantic Hybrid RAG grounded answer with source + 👍/👎
- *"Broken corridor light outside Blk 123"* -> Postal/OneMap Town Council lookup -> AI Vision photo inspection -> Ticket # + Trello Work Order card
- *"What's the best pizza place in Singapore?"* -> Refused cleanly & forwarded to staff review queue
- Staff group commands: `/progress <id> note`, `/resolve <id> note`, `/reply <user_id> text` (automatically notifies all subscribed neighbours and syncs Trello)

## Evaluation & Testing
- Run RAG benchmark (`Hit Rate @ 3`, `MRR`, Answer Accuracy, Refusal Accuracy):
  ```bash
  python eval.py
  ```
- Run automated unit & integration tests (including mocked Trello work-order dispatch):
  ```bash
  pytest test_suite.py -v
  ```
- Seed demo tickets for live presentations:
  ```bash
  python seed_demo.py
  ```
