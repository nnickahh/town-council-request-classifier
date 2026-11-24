"""Staff dashboard. Run: streamlit run dashboard.py
Reads the same SQLite database as the bot; replies/status changes are pushed to residents via the Telegram API."""
import asyncio
import json
import os
from datetime import date

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

import config
import db
import ingest
import pdf_converter
import policy_editor
import rag
import trello_client

st.set_page_config(page_title=f"{config.COUNCIL_NAME} · Operations", page_icon="🏛️", layout="wide")
db.init()

PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
if PASSWORD and not st.session_state.get("authed"):
    pw = st.text_input("Staff password", type="password")
    if pw and pw == PASSWORD:
        st.session_state["authed"] = True
        st.rerun()
    st.stop()

API = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}"
ICONS = {"open": "🔴", "in progress": "🟡", "resolved": "🟢"}
COLOR_MAP = {
    "open": [239, 68, 68, 200],       # Red
    "in progress": [245, 158, 11, 200], # Amber
    "resolved": [34, 197, 94, 200],   # Green
}


def tg_send(chat_id, text) -> bool:
    try:
        return requests.post(f"{API}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=10).ok
    except requests.RequestException:
        return False


@st.cache_data(ttl=3600, show_spinner=False)
def tg_photo(file_id):
    try:
        info = requests.get(f"{API}/getFile", params={"file_id": file_id}, timeout=10).json()
        path = info["result"]["file_path"]
        return requests.get(f"https://api.telegram.org/file/bot{config.TELEGRAM_TOKEN}/{path}", timeout=20).content
    except Exception:
        return None


def update_ticket(t, status, note):
    db.set_status(t["id"], status, note or None)
    if "trello_card_id" in t.keys() and t["trello_card_id"]:
        trello = trello_client.get_trello_client()
        if trello:
            trello.update_status_sync(t["trello_card_id"], status, note or None)
    text = f"Update on your report #{t['id']}: {status}." + (f"\nNote: {note}" if note else "")

    subs = []
    try:
        subs = json.loads(t["subscribers"] or "[]") if "subscribers" in t.keys() else []
    except Exception:
        pass
    if t["user_id"] not in subs:
        subs.append(t["user_id"])

    notified = 0
    for uid in subs:
        if tg_send(uid, text):
            notified += 1

    st.toast(f"Ticket #{t['id']} → {status} ({notified} resident(s) notified)")
    st.rerun()


def render():
    s = db.stats()
    rate = f"{s['answered'] / s['questions']:.0%}" if s["questions"] else "n/a"
    c = st.columns(5)
    c[0].metric("Tickets", s["tickets"])
    c[1].metric("Open", s["open"])
    c[2].metric("Resolved", s["resolved"])
    c[3].metric("Bot Answer Rate", rate)
    c[4].metric("👍 / 👎 Feedback", f"{s['up']} / {s['down']}")

    tab_m, tab_t, tab_q, tab_kb, tab_f = st.tabs([
        "🗺️ Live GIS Map",
        "🎫 Tickets Queue",
        "❓ Unanswered Questions",
        "📄 KB & Policy Copilot",
        "📊 Analytics & Insights",
    ])

    # ----------------- TAB 1: 3D GIS HEATMAP -----------------
    with tab_m:
        st.subheader("Singapore Estate Incident Map")
        all_tickets = db.list_tickets()
        if all_tickets:
            map_data = []
            for t in all_tickets:
                lat = t["lat"] if "lat" in t.keys() and t["lat"] else None
                lng = t["lng"] if "lng" in t.keys() and t["lng"] else None
                if lat and lng:
                    map_data.append({
                        "id": t["id"],
                        "category": t["category"],
                        "description": t["description"],
                        "location": t["location"] or "Unknown",
                        "status": t["status"],
                        "town_council": t["town_council"] or "General",
                        "lat": lat,
                        "lon": lng,
                        "color": COLOR_MAP.get(t["status"], [100, 100, 100, 200]),
                        "elevation": 50 if t["status"] == "resolved" else (150 if t["status"] == "in progress" else 300),
                    })

            if map_data:
                df_map = pd.DataFrame(map_data)
                layer = pdk.Layer(
                    "ColumnLayer",
                    data=df_map,
                    get_position=["lon", "lat"],
                    get_elevation="elevation",
                    elevation_scale=5,
                    radius=120,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                )
                scatter = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position=["lon", "lat"],
                    get_color="color",
                    get_radius=180,
                    pickable=True,
                )
                view_state = pdk.ViewState(
                    latitude=1.3521,
                    longitude=103.8198,
                    zoom=11,
                    pitch=45,
                    bearing=0,
                )
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer, scatter],
                        initial_view_state=view_state,
                        tooltip={"text": "Ticket #{id}: {category} ({status})\nCouncil: {town_council}\nLocation: {location}\n{description}"},
                    )
                )
                st.caption("🔴 Red = Open Fault | 🟡 Amber = In Progress | 🟢 Green = Resolved")
            else:
                st.info("No tickets with coordinates to plot yet.")
        else:
            st.info("No tickets in database.")

    # ----------------- TAB 2: TICKETS QUEUE -----------------
    with tab_t:
        f_col1, f_col2 = st.columns([2, 2])
        with f_col1:
            choice = st.selectbox("Status Filter", ["Needs attention", "Resolved", "All"], key="filter")
        statuses = {"Needs attention": ["open", "in progress"], "Resolved": ["resolved"], "All": None}[choice]
        tickets = db.list_tickets(statuses)
        if not tickets:
            st.info("Nothing here 🎉")
        for t in tickets:
            council_badge = f"🏛️ {t['town_council']} · " if ("town_council" in t.keys() and t["town_council"]) else ""
            sev = t["severity"] if "severity" in t.keys() and t["severity"] else "normal"
            sev_badge = "🔴 CRITICAL" if sev.lower() in {"high", "critical"} else ""
            title = f"{ICONS.get(t['status'], '⚪')} #{t['id']} · {council_badge}{t['category']} {sev_badge} · {t['description'][:50]}"
            with st.expander(title, expanded=t["status"] == "open"):
                left, right = st.columns([3, 2])
                with left:
                    st.write(f"**Issue Description:** {t['description']}")
                    if "ai_diagnosis" in t.keys() and t["ai_diagnosis"]:
                        st.info(f"🤖 **AI Vision Inspection:** {t['ai_diagnosis']}")
                    trello_url = t["trello_card_url"] if "trello_card_url" in t.keys() else None
                    if trello_url:
                        st.markdown(f"📋 **Trello Work Order:** [{trello_url}]({trello_url})")
                    elif trello_client.get_trello_client() is not None:
                        if st.button("📋 Submit Work Order to Trello", key=f"trello{t['id']}"):
                            trello = trello_client.get_trello_client()
                            try:
                                card = asyncio.run(
                                    trello.submit_work_order(
                                        ticket_id=t["id"],
                                        category=t["category"] or "other",
                                        description=t["description"],
                                        location=t["location"],
                                        town_council=t["town_council"] if "town_council" in t.keys() else None,
                                        severity=sev,
                                        ai_diagnosis=t["ai_diagnosis"] if "ai_diagnosis" in t.keys() else None,
                                        lat=t["lat"] if "lat" in t.keys() else None,
                                        lng=t["lng"] if "lng" in t.keys() else None,
                                    )
                                )
                                db.set_trello_card(t["id"], card.card_id, card.card_url)
                                st.toast(f"Submitted Ticket #{t['id']} to Trello ({card.list_name}) ✅")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not submit to Trello: {exc}")
                    st.caption(f"{council_badge}📍 {t['location'] or 'no location'} · created {t['created_at']} · resident id {t['user_id']}")
                    if "subscribers" in t.keys() and t["subscribers"]:
                        try:
                            num_subs = len(json.loads(t["subscribers"]))
                            if num_subs > 1:
                                st.caption(f"👥 **{num_subs} neighbours** subscribed to this ticket")
                        except Exception:
                            pass
                    note = st.text_input("Note to resident (optional)", key=f"note{t['id']}")
                    b1, b2 = st.columns(2)
                    if b1.button("🟡 In progress", key=f"p{t['id']}", disabled=t["status"] == "in progress"):
                        update_ticket(t, "in progress", note)
                    if b2.button("✅ Resolve", key=f"r{t['id']}", type="primary", disabled=t["status"] == "resolved"):
                        update_ticket(t, "resolved", note)
                with right:
                    if t["photo_id"]:
                        img = tg_photo(t["photo_id"])
                        if img:
                            st.image(img, caption=f"Resident upload #{t['id']}")

    # ----------------- TAB 3: UNANSWERED QUESTIONS -----------------
    with tab_q:
        st.caption("Resident questions not fully confident in knowledge base. Answers go to residents and can teach the bot.")
        qs = db.list_unanswered()
        if not qs:
            st.info("No open questions 🎉")
        for q in qs:
            with st.container(border=True):
                st.markdown(f"**#{q['id']}** {q['question']}")
                ans = st.text_area("Your answer", key=f"a{q['id']}")
                kb = st.checkbox("Also add to the bot's knowledge base", value=True, key=f"kb{q['id']}")
                if st.button("Send answer", key=f"s{q['id']}", type="primary", disabled=not ans.strip()):
                    sent = tg_send(q["user_id"], f"Message from {config.COUNCIL_NAME}:\n{ans.strip()}")
                    db.mark_handled(q["id"])
                    if kb:
                        import ingest
                        ingest.add_document(
                            f"Staff answer: {q['question'][:60]}",
                            date.today().isoformat(),
                            f"{q['question']}\n\n{ans.strip()}",
                            f"staff_answer_{q['id']}",
                        )
                    st.toast("Sent ✅" if sent else "Saved (could not message resident)")
                    st.rerun()

    # ----------------- TAB 4: KB INGESTION, SEMANTIC RAG & POLICY COPILOT -----------------
    with tab_kb:
        st.subheader("PDF Handbook Ingestion, Semantic RAG Tester & Living Policy Copilot")
        sub_rag, sub_pdf, sub_pol = st.tabs([
            "🔍 Semantic RAG FAQ Tester",
            "📥 PDF-to-Markdown Ingestion",
            "✍️ Policy Copilot (HITL)",
        ])

        with sub_rag:
            engine = rag.get_engine()
            n_chunks = engine.initialize()
            st.caption(
                f"Active backend: `{engine.embedding_provider.backend_name}` · "
                f"**{n_chunks}** semantic chunks across **{len(policy_editor.list_policy_documents())}** policy docs"
            )
            presets = [
                "How many bulky items can an HDB household remove for free each month?",
                "What is the 1.2m corridor fire safety rule?",
                "What are the permitted hours for noisy renovation drilling?",
                "What is the fine for feeding pigeons?",
                "How do I book a void deck for a funeral wake?",
                "What is the best pizza place in Singapore? (out-of-scope refusal test)",
            ]
            preset = st.selectbox("Quick FAQ test presets", ["(Type your own below)"] + presets, key="rag_preset")
            default_q = "" if preset == "(Type your own below)" else preset.replace(" (out-of-scope refusal test)", "")
            q_input = st.text_input("Ask a resident FAQ question", value=default_q, key="rag_query_input")
            if st.button("Run Semantic Hybrid RAG", type="primary", key="rag_run_btn") and q_input.strip():
                res = asyncio.run(rag.answer(q_input.strip()))
                if res["answer"] is None:
                    st.warning(
                        f"Refused (score={res['best_score']:.2f}, dist={res['best_distance']}) — "
                        "would escalate cleanly to Town Council staff."
                    )
                else:
                    st.success(f"Grounded Answer (score={res['best_score']:.2f}, model=`{res['model_used']}`)")
                    st.markdown(res["answer"])
                    if res["sources"]:
                        st.caption("Sources: " + " · ".join(f"{s['title']} ({s['updated']})" for s in res["sources"]))
                with st.expander("Retrieved Semantic Chunks & Breadcrumbs"):
                    for idx, ch in enumerate(res.get("retrieved_chunks", []), 1):
                        st.markdown(f"**{idx}.** `{ch['breadcrumb']}` (`{ch['source_file']}`)")

        with sub_pdf:
            st.caption("Upload a Town Council PDF circular or handbook to convert into Markdown and hot-reindex Semantic RAG.")
            uploaded = st.file_uploader("Upload PDF handbook (.pdf)", type=["pdf"], key="kb_pdf_upload")
            engine_choice = st.selectbox("Conversion engine", ["auto", "pypdf", "pymupdf", "gemini"], key="kb_pdf_engine")
            if uploaded is not None and st.button("Convert PDF → Markdown & Hot Re-Index", type="primary", key="kb_pdf_btn"):
                try:
                    out_md, conv = pdf_converter.ingest_pdf_bytes_to_kb(
                        uploaded.getvalue(),
                        uploaded.name,
                        policy_editor.DOCS_DIR,
                        engine=engine_choice,
                    )
                    chunks = rag.get_engine().reindex()
                    st.success(
                        f"Converted `{uploaded.name}` → `{out_md.name}` "
                        f"({conv.page_count} pages, {conv.table_count} tables, {conv.word_count} words via `{conv.engine_used}`). "
                        f"Semantic RAG hot-reindexed ({chunks} chunks)!"
                    )
                    with st.expander("Preview Converted Markdown"):
                        st.code(conv.markdown_text[:3000], language="markdown")
                except Exception as exc:
                    st.error(f"PDF conversion failed: {exc}")

            st.markdown("**Indexed Knowledge Base Documents (`data/docs/`):**")
            for doc in policy_editor.list_policy_documents():
                st.caption(f"• `{doc.name}` ({doc.stat().st_size:,} bytes)")

        with sub_pol:
            st.caption("Amend Town Council policy documents in natural language with visual diff review, audit logs, and 1-click rollback.")
            docs = policy_editor.list_policy_documents()
            if not docs:
                st.info("No policy documents found in data/docs/.")
            else:
                doc_names = [d.name for d in docs]
                sel_name = st.selectbox("Target policy document", doc_names, key="pol_doc_select")
                sel_path = next(d for d in docs if d.name == sel_name)
                doc_text = sel_path.read_text(encoding="utf-8")
                sections = policy_editor.parse_handbook_sections(doc_text)

                intent_text = st.text_input(
                    "Describe the policy change in plain English",
                    placeholder="e.g. Increase refundable security deposit from $200 to $300",
                    key="pol_intent_input",
                )
                author_name = st.text_input("Officer name", value="Council Officer", key="pol_author")

                if st.button("Draft Surgical Amendment", type="primary", key="pol_draft_btn") and intent_text.strip():
                    target_sec, score, rationale = policy_editor.locate_target_section(sections, intent_text.strip())
                    proposal = policy_editor.generate_policy_amendment(target_sec, intent_text.strip())
                    st.session_state["pol_proposal"] = proposal
                    st.session_state["pol_doc_path"] = str(sel_path)
                    st.session_state["pol_rationale"] = rationale

                proposal = st.session_state.get("pol_proposal")
                if proposal and st.session_state.get("pol_doc_path") == str(sel_path):
                    st.info(f"🎯 **Localized Section:** `{proposal.section.breadcrumb}` — {st.session_state.get('pol_rationale', '')}")
                    st.markdown(
                        f"**Summary ({proposal.model_used}):** {proposal.explanation} "
                        f"(+{proposal.additions_count} / -{proposal.deletions_count} lines)"
                    )
                    st.markdown(policy_editor.render_diff_html(proposal.diff_lines), unsafe_allow_html=True)

                    c_apply, c_clear = st.columns(2)
                    if c_apply.button("✅ Approve, Apply & Hot Re-Index", type="primary", key="pol_apply_btn"):
                        ok, rec, msg = policy_editor.apply_amendment_to_handbook(
                            sel_path,
                            proposal.section,
                            proposal.proposed_raw_block,
                            proposal.intent,
                            proposal.explanation,
                            author=author_name or "Council Officer",
                        )
                        st.session_state.pop("pol_proposal", None)
                        st.toast(msg)
                        st.rerun()
                    if c_clear.button("Discard Draft", key="pol_discard_btn"):
                        st.session_state.pop("pol_proposal", None)
                        st.rerun()

                history = policy_editor.load_audit_history(limit=10)
                if history:
                    st.divider()
                    st.markdown("**Recent Amendment Audit Trail & 1-Click Rollback**")
                    for item in history:
                        with st.container(border=True):
                            st.markdown(
                                f"**[{item['status']}]** `{item['document_name']}` → `{item['target_section']}` "
                                f"by *{item['author']}* ({item['timestamp'][:19]} UTC)"
                            )
                            st.caption(f"Intent: {item['intent']} · Backup: `{item['backup_file']}`")
                            if item["status"] == "APPLIED":
                                target_doc = policy_editor.DOCS_DIR / item["document_name"]
                                if st.button("↩️ Rollback to this snapshot", key=f"rb_{item['log_id']}"):
                                    ok, msg = policy_editor.rollback_amendment(target_doc, item["backup_file"])
                                    st.toast(msg)
                                    st.rerun()

    # ----------------- TAB 5: ANALYTICS -----------------
    with tab_f:
        cols = st.columns(3)
        all_t = db.list_tickets()
        df_t = pd.DataFrame([dict(t) for t in all_t]) if all_t else pd.DataFrame()
        with cols[0]:
            st.subheader("Tickets by Category")
            if not df_t.empty and "category" in df_t:
                st.bar_chart(df_t["category"].value_counts())
        with cols[1]:
            st.subheader("Tickets by Town Council")
            if not df_t.empty and "town_council" in df_t:
                st.bar_chart(df_t["town_council"].value_counts())
        with cols[2]:
            st.subheader("Citizen Downvotes (👎)")
            bad = db.list_downvoted()
            if not bad:
                st.caption("None yet.")
            for b in bad:
                st.markdown(f"**Q:** {b['question']}\n\n**A:** {b['answer']}")
                st.divider()


st.title(f"🏛️ {config.COUNCIL_NAME} · Staff Operations Command")
live = st.sidebar.toggle("Live Refresh (10s)", value=True)
st.sidebar.caption("Turn off while typing long replies.")
st.fragment(run_every=10 if live else None)(render)()
