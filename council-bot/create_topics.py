import json
import time
import requests
import config
import db

db.init()

API = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}"

# 17 Singapore Town Councils in clean lowercase
SINGAPORE_TOWN_COUNCILS = [
    ("Ang Mo Kio Town Council", "ang-mo-kio"),
    ("Bishan-Toa Payoh Town Council", "bishan-toa-payoh"),
    ("Tampines Town Council", "tampines"),
    ("Jurong-Clementi Town Council", "jurong-clementi"),
    ("Aljunied-Hougang Town Council", "aljunied-hougang"),
    ("East Coast Town Council", "east-coast"),
    ("Holland-Bukit Panjang Town Council", "holland-bukit-panjang"),
    ("Jalan Besar Town Council", "jalan-besar"),
    ("Marine Parade Town Council", "marine-parade"),
    ("Marsiling-Yew Tee Town Council", "marsiling-yew-tee"),
    ("Nee Soon Town Council", "nee-soon"),
    ("Pasir Ris-Punggol Town Council", "pasir-ris-punggol"),
    ("Sembawang Town Council", "sembawang"),
    ("Sengkang Town Council", "sengkang"),
    ("Tanjong Pagar Town Council", "tanjong-pagar"),
    ("West Coast Town Council", "west-coast"),
    ("Chua Chu Kang Town Council", "chua-chu-kang"),
]

def main():
    chat_id = config.STAFF_CHAT_ID
    if not chat_id:
        print("STAFF_CHAT_ID not configured.")
        return

    print(f"Creating topics in group {chat_id}...")
    success = 0
    for full_name, slug in SINGAPORE_TOWN_COUNCILS:
        existing = db.get_topic(full_name)
        if existing:
            print(f"[EXISTS] {slug} already mapped to thread {existing}")
            continue

        resp = requests.post(
            f"{API}/createForumTopic",
            json={"chat_id": chat_id, "name": slug},
            timeout=10,
        )
        data = resp.json()
        if data.get("ok"):
            tid = data["result"]["message_thread_id"]
            db.save_topic(full_name, tid)
            print(f"[OK] Created topic: #{slug} (thread_id: {tid})")
            success += 1
            time.sleep(0.5)  # Telegram rate limit safety
        else:
            err = data.get("description", "")
            print(f"[ERROR] Failed to create #{slug}: {err}")
            if "not enough rights" in err:
                print("\n⚠️ The bot needs to be an Admin with 'Manage Topics' permission enabled in group settings!")
                break

    print(f"\nDone! Created {success} topics.")

if __name__ == "__main__":
    main()
