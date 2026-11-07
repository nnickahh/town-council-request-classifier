"""Fills the database with demo data so the dashboard isn't empty. Run once before presenting: python seed_demo.py"""
import council_lookup
import db

db.init()
demo = [
    (1, "lighting", "Streetlight flickering and then going out every night", "560123", "open", "Flickering sodium vapor bulb", "Medium"),
    (2, "roads", "Large pothole damaging cars near the carpark entrance", "Blk 201 Tampines St 21 520201", "in progress", "Asphalt surface cavity (~10cm)", "High"),
    (3, "waste", "Bulky discarded sofa blocking corridor staircase", "Blk 104 Toa Payoh Lorong 1 310104", "open", "Combustible furniture obstruction", "High"),
    (4, "noise", "Unauthorized renovation drilling before 8am", "Blk 512 Jurong West St 52 640512", "resolved", "Noise disturbance outside allowed hours", "Low"),
    (5, "parks", "Broken swing chain in the children playground", "Bedok Reservoir Park 479224", "resolved", "Fractured stainless steel swing link", "High"),
    (6, "lighting", "Dark staircase landing between 4th and 5th floor", "Blk 802 Woodlands St 81 730802", "open", "Dead LED batten luminaire", "Medium"),
    (7, "water", "Ceiling drainage pipe dripping in void deck", "Blk 25 Sengkang West Ave 792025", "in progress", "PVC rainwater downpipe joint seepage", "Medium"),
    (8, "waste", "Central rubbish chute hopper door jammed", "Blk 63 Queenstown Strathmore Ave 140063", "open", "Overstuffed refuse choke", "High"),
]

for uid, cat, desc, loc, status, diag, sev in demo:
    resolved_loc, council, lat, lng = council_lookup.resolve_town_council(loc)
    tid = db.create_ticket(
        uid, cat, desc, resolved_loc, None,
        town_council=council, lat=lat, lng=lng,
        ai_diagnosis=diag, severity=sev
    )
    if status != "open":
        db.set_status(tid, status, None)

for uid, q, a, src, ok, fb in [
    (1, "How many bulky items can I throw for free each month?", "Up to 3 movable household items per month for HDB residents.", "Bulky Item Removal Service (updated 2026-03)", 1, 1),
    (2, "What are the allowed drilling and renovation hours?", "Mondays to Saturdays, 9:00 AM to 6:00 PM. No noisy works on Sundays/PH.", "Renovation Regulations (updated 2026-03)", 1, 1),
    (3, "Can I put shoe racks along the corridor?", "Yes, provided a minimum clear escape passage of 1.2 metres is maintained.", "Common Corridor Safety (updated 2026-03)", 1, 1),
    (4, "What is the penalty for feeding pigeons?", "Under the Wildlife Act, fines range up to $5,000 for first offence.", "Bird Feeding Guidelines (updated 2026-03)", 1, 1),
]:
    lid = db.log_qa(uid, q, a, src, ok)
    if fb is not None:
        db.set_feedback(lid, fb)

print("Demo data successfully populated with Singapore coordinates, AI diagnoses, and Town Councils.")
