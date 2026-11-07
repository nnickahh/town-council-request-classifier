# Hackathon demo kit

## One-line pitch
"A Telegram assistant that answers residents from official council documents, never guesses, and turns problem reports into tickets that staff close from a live dashboard."

## Problem / solution (30 seconds)
- Councils get the same questions every day, and reports get lost in phone calls and emails.
- Residents are already on Telegram, so there's no app to install.
- Answers come only from council documents, with a source and date. If the bot isn't sure, it hands off to a human.

## 4-minute live demo
1. **Question**: ask "How much is a parking permit?" and show the answer with its source line. Tap 👍.
2. **Honesty**: ask "Can I bring my dog to the council meeting?" and the bot says it isn't sure and escalates.
3. **Report**: type "There's a pothole on Oak Avenue", give a location, send a photo, and show the ticket number.
4. **Dashboard**: the ticket (with photo) is already there. In "Unanswered questions", type an answer and tick "add to knowledge base".
5. **Close the loop**: click Resolve and show the resident's phone receiving the update.
6. **Learning**: ask the dog question again and the bot now answers it from the staff answer.
7. **Numbers**: show the `eval.py` score ("answers X% correctly, refuses off-topic questions").

## Slides (5 max)
1. Problem  2. Solution (screenshot of bot)  3. Architecture (Telegram -> intent router -> RAG / ticketing -> dashboard)  4. Results (eval score, answer rate)  5. Next steps (multilingual, voice, integration with council CRM)

## Likely judge questions
- **Hallucinations?** Answers use only retrieved text, there's a distance cutoff, and the bot says "I'm not sure" otherwise. The eval set measures refusals as well as correct answers.
- **Privacy?** We store only what's needed (messages, tickets), tell users up front, and staff-only access is behind a group and an optional dashboard password.
- **Emergencies?** Keyword and LLM detection reply instantly with emergency numbers and skip the RAG step.
- **Cost?** A small model routes intent and a stronger one answers; per-user rate limiting caps usage.
- **Scale?** Swap SQLite for Postgres and Chroma for pgvector or Qdrant; the architecture stays the same.

## Before you present
- `python ingest.py` (use realistic documents), then `python seed_demo.py`
- Run `python bot.py` and `streamlit run dashboard.py` in two terminals
- Test the whole demo once on your own phone, and keep screenshots as a backup in case Wi-Fi fails
