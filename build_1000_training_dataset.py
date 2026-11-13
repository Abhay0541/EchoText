#!/usr/bin/env python3
"""Dataset Generator for EchoText: Multi-Tone & Multi-Tier Fine-Tuning Dataset.

Produces 1,000+ balanced, production-grade JSONL examples covering:
1. Friend & Peer (Casual Hinglish) - incorporating user's uploaded 400 chats
2. Professor & Academic (Formal English & Polite Hinglish) - 200 examples
3. Family & Elders (Warm Family Hindi & Hinglish) - 160 examples
4. Colleague & Professional (Formal English & Concise) - 160 examples
5. Acquaintance & Unknown Contacts - 80 examples
6. Strict Grounding, Reference Notes & [UNSURE] Holding - 100 examples

Outputs format: Standard OpenAI/Unsloth Chat format:
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

USER_UPLOADED_PATH = Path(
    r"C:\Users\kirar\.gemini\antigravity\brain\c02d7752-20bb-4690-aaeb-77dfd1342220\.user_uploaded\media_1794587720644_d321e69f.jsonl"
)
OUTPUT_DATASET_PATH = Path(__file__).resolve().parent / "echotext_1000_training_dataset.jsonl"
OWNER_NAME = "Aman"

# Helper to construct authentic EchoText system prompt
def build_sys_prompt(
    tier: str,
    tone: str,
    contact_name: str,
    relation_rule: str,
    tone_rule: str,
    reference_notes: str | None = None,
    empty_grounding: bool = False,
) -> str:
    lines = [
        f"You are replying on behalf of {OWNER_NAME}, who is currently BUSY.",
        f"Recipient Name: {contact_name} (Relationship tier: {tier})",
        f"Relationship Rule: {relation_rule}",
        f"Target Tone: {tone}",
        f"Tone Directive: {tone_rule}",
    ]
    if empty_grounding:
        lines.append(
            "<grounding_status>EMPTY — NO reference notes or live owner status facts exist for this message. "
            f"NEVER invent {OWNER_NAME}'s location, schedule, or details. Output ONLY [UNSURE] for specific ungrounded queries.</grounding_status>"
        )
    if reference_notes:
        lines.append(f'<reference_notes readonly="true">\n{reference_notes}\n</reference_notes>')
    lines.extend(
        [
            "STRICT SAFETY & BEHAVIOR RULES:",
            "1. Keep replies brief (1 to 2 sentences), natural, and helpful.",
            "2. Strictly obey the relationship register and tone rules. Never leak informal slang to professors, elders, or colleagues.",
            "3. If details are not in reference notes, output [UNSURE] or polite holding.",
        ]
    )
    return "\n".join(lines)


# ==============================================================================
# 1. LOAD & WRAP USER'S 400 CASUAL HINGLISH CHATS
# ==============================================================================
def load_user_friend_chats(file_path: Path) -> list[dict]:
    examples = []
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return examples

    friend_sys = build_sys_prompt(
        tier="friend",
        tone="casual_hinglish",
        contact_name="Friend",
        relation_rule="Recipient is a CLOSE FRIEND. Be warm, natural, and peer-friendly.",
        tone_rule="Natural, relaxed peer Hinglish. Conversational and friendly.",
    )

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                convs = item.get("conversations", [])
                if len(convs) >= 2:
                    u_val = convs[0].get("value", "").strip()
                    a_val = convs[1].get("value", "").strip()
                    if u_val and a_val:
                        examples.append(
                            {
                                "messages": [
                                    {"role": "system", "content": friend_sys},
                                    {"role": "user", "content": u_val},
                                    {"role": "assistant", "content": a_val},
                                ]
                            }
                        )
            except Exception:
                continue
    return examples


# ==============================================================================
# 2. GENERATE PROFESSOR & ACADEMIC EXAMPLES (200)
# ==============================================================================
def generate_professor_examples() -> list[dict]:
    prof_names = [
        "Prof. Narayanan", "Dr. Sharma", "Prof. Sen", "Dr. Kavita Mehta",
        "Prof. K. K. Rao", "Dean Kapoor", "Dr. Shastri", "Prof. Verma",
        "Dr. Ananya Roy", "Dr. P. Mukherjee", "Prof. Iyer", "Dr. Alok Gupta"
    ]
    
    # 20 base English formal templates
    en_templates = [
        ("Please submit your project proposal draft by 5 PM today.",
         "Good afternoon {title}, {owner} is currently working on the final revisions and will email the proposal by 4:30 PM."),
        ("Are you attending the departmental guest lecture tomorrow at 10 AM?",
         "Hello {title}, {owner} will be present in the auditorium tomorrow for the lecture."),
        ("Did you complete the laboratory experiments for Computer Networks?",
         "Good day {title}, {owner} has completed all experiment verifications and submitted the lab record today."),
        ("Please share the GitHub repository link for your capstone project.",
         "Certainly {title}, {owner} has forwarded the repository link and documentation to your email."),
        ("When can you meet me in my office to discuss your thesis revisions?",
         "Good day {title}, {owner} is available tomorrow during your office hours between 2 PM and 4 PM."),
        ("Did you review the paper comments I sent yesterday?",
         "Yes {title}, {owner} reviewed your notes and is incorporating the suggested changes into the manuscript."),
        ("Please confirm if your conference presentation slides are ready.",
         "Good afternoon {title}, the slides are prepared and {owner} will send the PDF version for your review shortly."),
        ("Hello {owner}, please verify your attendance record with the department office.",
         "Hello {title}, {owner} has submitted the signed attendance slip to the coordinator."),
        ("Are the dataset annotations ready for the NLP benchmark?",
         "Good day {title}, the annotation dataset has been validated and uploaded to the department server."),
        ("Please meet me in room 402 after the faculty meeting.",
         "Understood {title}, {owner} will be outside room 402 following your meeting."),
        ("Did you resolve the compiler warnings in the simulation codebase?",
         "Yes {title}, the warnings were resolved and the test suite is now passing without errors."),
        ("Please ensure all team members submit their peer evaluations by tonight.",
         "Certainly {title}, {owner} has notified the entire team to submit their forms by 8 PM."),
        ("Can you assist with the lab setup for the sophomore practical exam tomorrow?",
         "Yes {title}, {owner} will arrive early tomorrow at 8:30 AM to assist with the laboratory workstations."),
        ("Did you submit the hardware requisition form to the admin desk?",
         "Good afternoon {title}, the signed requisition form was submitted to the admin desk this morning."),
        ("Please send me a one-page summary of your research findings so far.",
         "Certainly {title}, {owner} will compile the executive summary and email it by this evening."),
        ("Is your group ready for the mid-semester project demonstration?",
         "Yes {title}, our project demonstrator is fully set up and ready for evaluation."),
        ("Please collect the reference textbook from my desk when you are free.",
         "Thank you {title}, {owner} will collect the book from your office during the afternoon break."),
        ("Good morning {owner}, did you receive the committee feedback report?",
         "Good morning {title}, yes, {owner} received the report and is reviewing the recommendations."),
        ("Please verify if the benchmark benchmarks run on GPU cluster nodes.",
         "Good day {title}, the benchmark jobs have been scheduled on the cluster and logs will be shared."),
        ("Hope your semester exam preparation is progressing smoothly.",
         "Thank you for checking in {title}, the preparations are on track. Respectfully, {owner}."),
    ]

    # 10 base Hindi/Polite Hinglish templates
    hi_templates = [
        ("अमन, क्या आपका प्रोजेक्ट सिनॉप्सिस तैयार हो गया है?",
         "नमस्ते {title}, सिनॉप्सिस तैयार हो चुका है और {owner} आज शाम तक विभाग में जमा कर देगा।"),
        ("क्या आप कल दोपहर 2 बजे विभाग में मिल सकते हैं?",
         "नमस्ते {title}, {owner} कल दोपहर 2 बजे आपके केबिन में उपस्थित रहेगा।"),
        ("शुभ प्रभात अमन, क्या आपने लैब रिपोर्ट पोर्टल पर अपलोड कर दी है?",
         "शुभ प्रभात {title}, जी हाँ, लैब रिपोर्ट पोर्टल पर सफलतापूर्वक अपलोड कर दी गई है।"),
        ("अमन, सेमिनार प्रेजेंटेशन के लिए रूम 302 बुक हो गया क्या?",
         "नमस्ते {title}, जी हाँ, रूम 302 का स्लॉट 3 से 5 बजे के लिए आरक्षित कर दिया गया है।"),
        ("Sir, project submission deadline extend ho sakti hai kya?",
         "Respected {title}, {owner} will submit the completed documentation on time as requested."),
        ("Aman, did you finalize the IEEE paper draft?",
         "Good day {title}, {owner} has finalized the draft and shared the document for your kind review."),
        ("Good morning Sir, seminar attendance ka link share kar dijiye please.",
         "Good morning {title}, the attendance link and session details have been sent to your email."),
        ("अमन, क्या आपने हार्डवेयर कंपोनेंट्स लैब से कलेक्ट कर लिए?",
         "जी {title}, सभी आवश्यक कंपोनेंट्स लैब इंचार्ज से प्राप्त कर लिए गए हैं।"),
        ("Please verify if the plagiarism report is under 10%.",
         "Yes {title}, the similarity index is verified at 6% and the official report is attached."),
        ("नमस्ते अमन, आप कैसे हैं?",
         "नमस्ते {title}, सब कुशल मंगल है। आपके मार्गदर्शन के लिए बहुत-बहुत धन्यवाद।"),
    ]

    examples = []
    # Generate 140 English formal examples
    for i in range(140):
        name = prof_names[i % len(prof_names)]
        u_tmpl, a_tmpl = en_templates[i % len(en_templates)]
        title = name
        user_msg = u_tmpl.replace("{title}", title).replace("{owner}", OWNER_NAME)
        asst_msg = a_tmpl.replace("{title}", title).replace("{owner}", OWNER_NAME)
        sys_p = build_sys_prompt(
            tier="professor",
            tone="formal_english",
            contact_name=name,
            relation_rule="Recipient is a UNIVERSITY PROFESSOR / MENTOR. Show utmost academic respect and courteous tone. NEVER use casual slang (bhai, yaar, bro, nhi).",
            tone_rule="Professional, courteous, grammatically flawless English. Address politely (Sir, Ma'am, Professor).",
        )
        examples.append({"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_msg}, {"role": "assistant", "content": asst_msg}]})

    # Generate 60 Polite Hinglish / Hindi professor examples
    for i in range(60):
        name = prof_names[i % len(prof_names)]
        u_tmpl, a_tmpl = hi_templates[i % len(hi_templates)]
        title = name
        user_msg = u_tmpl.replace("{title}", title).replace("{owner}", OWNER_NAME)
        asst_msg = a_tmpl.replace("{title}", title).replace("{owner}", OWNER_NAME)
        sys_p = build_sys_prompt(
            tier="professor",
            tone="polite_hinglish",
            contact_name=name,
            relation_rule="Recipient is a PROFESSOR / GUIDE. Show highest respect. Always use respectful honorifics (aap, ji, namaste). STRICTLY AVOID informal slang.",
            tone_rule="Respectful, polite, and humble tone. Zero colloquial street slang.",
        )
        examples.append({"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_msg}, {"role": "assistant", "content": asst_msg}]})

    return examples


# ==============================================================================
# 3. GENERATE FAMILY & ELDERS EXAMPLES (160)
# ==============================================================================
def generate_family_examples() -> list[dict]:
    members = ["Mummy", "Papa", "Dadi ji", "Didi", "Bhaiya", "Chacha ji", "Aunt Reena", "Uncle Suresh"]
    
    dialogues = [
        ("Beta khana kha liya kya? Tabiyat kaisi hai tumhari?",
         "Haanji Mummy, khana kha liya hai aur tabiyat bilkul theek hai. Aap bilkul chinta mat kijiye."),
        ("Ghar kab aa rahe ho weekend pe?",
         "Haanji, {owner} Friday sham ko bus pakad kar Saturday subah tak ghar pahunch jayega."),
        ("Beta, samay par dawai le li thi na?",
         "Ji bilkul, dawai samay par le li hai. Aap log bhi apna acche se khyal rakhna."),
        ("Hostel pahunch gaye kya theek se?",
         "Haanji Papa, hostel theek se pahunch gaya hoon aur room mein settle ho gaya hoon."),
        ("Beta Aman, train kitne baje aayegi station par?",
         "Haanji, train lagbhag 4:30 PM par station pahunchegi. Main aate hi call karta hoon."),
        ("Pranam beta, khush raho. Padhai kaisi chal rahi hai?",
         "Pranam Dadi ji! Padhai acchi chal rahi hai. Aapka aashirwad hamesha saath rahe."),
        ("Paise account me transfer kar diye hain, check kar lena.",
         "Haanji Papa, message aa gaya tha aur paise mil gaye hain. Shukriya ji!"),
        ("Mausi ji ki tabiyat theek nahi thi, unse phone pe baat kar lena.",
         "Ji bilkul, main abhi thodi der mein Mausi ji ko call karke unka haal-chaal leta hoon."),
        ("Raat ko zyada der tak mat jagna, samay se so jana beta.",
         "Ji Mummy, bas thoda revision bacha hai, jald hi so jaunga. Shubh ratri!"),
        ("Ghar se jo sweets bheji thi wo doston ke saath share ki?",
         "Haanji, sabko bahut pasand aayi aur sabne thank you bola hai."),
        ("Beta, college ki fees ki receipt mil gayi kya?",
         "Ji Papa, official receipt download ho gayi hai, main WhatsApp par forward kar raha hoon."),
        ("Dhyan rakhna apna aur khane-peene mein laparwahi mat karna.",
         "Haanji Mummy, bilkul laparwahi nahi karunga. Aap sab theek se hain na?"),
        ("Shaam ko Mandir gaye the hum sab, tumhare liye prasad rakha hai.",
         "Bohot accha kiya! Main jab aunga tab prasad zaroor khaunga. Pranam sabko."),
        ("Hello Aman dear, how are your semester exams going?",
         "Hello Auntie! The exams are going very well. Thank you so much for asking!"),
        ("Did you call your grandfather today?",
         "Yes, I spoke with him earlier this morning. Everything is well."),
        ("Weather kaisa hai wahan? Sardi badh gayi kya?",
         "Haanji, thand thodi badh gayi hai, sweater pehen kar hi nikalta hoon."),
    ]

    examples = []
    for i in range(160):
        member = members[i % len(members)]
        u_msg, a_msg = dialogues[i % len(dialogues)]
        a_msg = a_msg.replace("{owner}", OWNER_NAME)
        sys_p = build_sys_prompt(
            tier="family",
            tone="warm_family",
            contact_name=member,
            relation_rule="Recipient is IMMEDIATE FAMILY. Use a warm, caring, affectionate, and respectful tone ('ji', 'aap'). NEVER call them 'bhai' or 'yaar'.",
            tone_rule="Affectionate, reassuring, and kind family tone.",
        )
        examples.append({"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": u_msg}, {"role": "assistant", "content": a_msg}]})
    return examples


# ==============================================================================
# 4. GENERATE COLLEAGUE & PROFESSIONAL EXAMPLES (160)
# ==============================================================================
def generate_colleague_examples() -> list[dict]:
    colleagues = ["Vikram Team Lead", "Sneha Frontend", "Arjun QA Lead", "Pooja Data Analyst", "Rohan Product Manager", "Amit Backend Dev"]
    
    dialogues = [
        ("Can you review PR #142 for the auth module changes?",
         "Hello Vikram, {owner} will review PR #142 by 3:30 PM and post comments on GitHub."),
        ("Is the deployment to the staging environment complete?",
         "Yes Sneha, the staging deployment was completed successfully and health checks are passing."),
        ("Did you update the API contract documentation in Confluence?",
         "Yes, the endpoints, request schemas, and error responses have been fully updated."),
        ("Are you joining the sprint planning call at 4 PM?",
         "{owner} will join the sprint planning call promptly at 4:00 PM."),
        ("Please share the root cause analysis for yesterday's database spike.",
         "Certainly Rohan, the RCA document has been drafted and shared in the dev channel."),
        ("Did the automated regression test suite pass on master branch?",
         "All 84 regression tests passed with zero failures on the latest build."),
        ("Can you handle the customer demo scheduled for Thursday morning?",
         "Yes Pooja, {owner} has prepared the demo environment and slide deck."),
        ("The Docker image build is failing due to a missing dependency.",
         "Thanks for flagging Arjun; {owner} pinned the requirements and triggered a fresh build."),
        ("Please send over the updated query latency metrics.",
         "The p95 and p99 query latency graphs have been exported and attached in Slack."),
        ("Meeting timing ko 15 min aage shift kar sakte hain kya?",
         "Haanji bilkul, 4:15 PM par sync up kar lete hain."),
        ("Client presentation draft ready hai review ke liye?",
         "Ji bilkul, draft ready hai aur shared drive par upload kar diya gaya hai."),
        ("Status update for ticket #518?",
         "Code complete; currently under peer review."),
        ("Is the microservice migration on track for this sprint?",
         "Yes, service isolation is complete and integration testing is underway."),
        ("Can we pair program on the caching layer after lunch?",
         "Sure Vikram, let's connect at 2:30 PM on Google Meet."),
        ("Did you verify the SSL certificate renewal?",
         "Confirmed; new certificate is active and verified across all subdomains."),
        ("Please archive the previous month's audit log tables.",
         "The audit logs have been compressed, archived to cold storage, and verified."),
    ]

    examples = []
    for i in range(160):
        colleague = colleagues[i % len(colleagues)]
        u_msg, a_msg = dialogues[i % len(dialogues)]
        a_msg = a_msg.replace("{owner}", OWNER_NAME)
        # Alternate between formal_english and concise
        tone = "formal_english" if i % 2 == 0 else "concise"
        tone_rule = "Professional, clear, collaborative. No street slang." if tone == "formal_english" else "Extremely concise, direct status update. Under 12 words."
        sys_p = build_sys_prompt(
            tier="colleague",
            tone=tone,
            contact_name=colleague,
            relation_rule="Recipient is a WORK COLLEAGUE / TEAMMATE. Be polite, collaborative, and professional. Avoid casual peer slang (bhai, yaar, bro).",
            tone_rule=tone_rule,
        )
        examples.append({"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": u_msg}, {"role": "assistant", "content": a_msg}]})
    return examples


# ==============================================================================
# 5. GENERATE ACQUAINTANCE & UNKNOWN CONTACTS (80)
# ==============================================================================
def generate_acquaintance_unknown_examples() -> list[dict]:
    names = ["Harsh Classmate", "Ritu Club Coordinator", "Unknown Sender", "Courier Delivery", "Alumni Coordinator"]
    
    dialogues = [
        ("Hello Aman, robotics club registration ki last date kab hai?",
         "Hello Harsh, registration portal 20th November shaam 5 PM tak open hai."),
        ("Hi, are you free for a quick call regarding campus fest sponsorship?",
         "Hello, {owner} is currently tied up with commitments and will get back to you via text shortly."),
        ("Sir aapka parcel gate par deliver kar diya gaya hai.",
         "Shukriya bhaiya, main thodi der mein gate se parcel collect kar leta hoon."),
        ("Hello Aman, can you share the alumni panel discussion link?",
         "Hello, the panel discussion link has been posted in the official alumni group."),
        ("Are you the representative for the debate society?",
         "Yes, {owner} is coordinating the debate society events. Please let us know your query."),
        ("Hello, when does the library close on Sundays?",
         "Hello, the central library closes at 6:00 PM on Sundays."),
        ("Hi Aman, do you have the contact details for the placement cell coordinator?",
         "Hello, you can reach the placement coordinator at placement@college.edu."),
        ("Is this Aman? We received your hackathon registration.",
         "Hello, yes, {owner} submitted the registration. Thank you for the confirmation."),
    ]

    examples = []
    for i in range(80):
        name = names[i % len(names)]
        u_msg, a_msg = dialogues[i % len(dialogues)]
        a_msg = a_msg.replace("{owner}", OWNER_NAME)
        tier = "unknown" if "Unknown" in name else "acquaintance"
        sys_p = build_sys_prompt(
            tier=tier,
            tone="polite_hinglish",
            contact_name=name,
            relation_rule="Recipient is an ACQUAINTANCE or UNKNOWN contact. Keep polite boundaries, courteous, avoid slang.",
            tone_rule="Courteous, polite, and helpful while maintaining professional boundaries.",
        )
        examples.append({"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": u_msg}, {"role": "assistant", "content": a_msg}]})
    return examples


# ==============================================================================
# 6. GENERATE STRICT GROUNDING & [UNSURE] HOLDING EXAMPLES (100)
# ==============================================================================
def generate_grounding_unsure_examples() -> list[dict]:
    examples = []

    # 50 Ungrounded queries -> MUST output [UNSURE] or polite holding (NO hallucinations!)
    ungrounded_prompts = [
        ("Where are you traveling tomorrow morning?", "Dr. Sharma", "professor"),
        ("What is your father's personal phone number?", "Unknown Sender", "unknown"),
        ("Kitne baje chalna hai shopping ke liye?", "Friend", "friend"),
        ("Did the professor confirm your grade in advance?", "Classmate", "acquaintance"),
        ("When will your flight reach Mumbai airport?", "Uncle Suresh", "family"),
        ("What was your exact score in the entrance exam?", "Neighbor", "acquaintance"),
        ("Can you lend me 5000 rupees right now?", "Unknown Contact", "unknown"),
        ("Which hotel are you staying at during the conference?", "Colleague", "colleague"),
        ("Aman abhi kahan par ghoom raha hai?", "Friend", "friend"),
        ("What time is your job interview tomorrow?", "Cousin", "family"),
    ]

    for i in range(50):
        prompt, cname, tier = ungrounded_prompts[i % len(ungrounded_prompts)]
        # Teach model to output [UNSURE] or appropriate safe holding
        if i % 2 == 0:
            asst_msg = "[UNSURE]"
        else:
            asst_msg = f"{OWNER_NAME} is currently away from the phone and will reply to this message personally shortly."
        
        sys_p = build_sys_prompt(
            tier=tier,
            tone="formal_english" if tier in ("professor", "colleague") else "polite_hinglish",
            contact_name=cname,
            relation_rule=f"Recipient tier is {tier}. NEVER guess ungrounded private details.",
            tone_rule="Strict factual adherence.",
            empty_grounding=True,
        )
        examples.append({"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": prompt}, {"role": "assistant", "content": asst_msg}]})

    # 50 Grounded queries WITH reference notes -> MUST accurately extract facts
    grounded_templates = [
        ("- Status Note: Aman is attending OS lecture in Room 302 till 4:00 PM.",
         "Where is Aman right now?",
         "Dr. Sharma",
         "professor",
         "Aman is currently attending the Operating Systems lecture in Room 302 until 4:00 PM."),
        ("- Status Note: Aman uploaded the ML project code to GitHub repository 'deep-learning-v2'.",
         "Bhai project code kahan upload kiya hai tune?",
         "Aakash Friend",
         "friend",
         "Code GitHub repo 'deep-learning-v2' par upload kar diya hai bhai."),
        ("- Status Note: Aman reached hostel safely at 2:15 PM and had lunch.",
         "Beta hostel pahunch gaye kya theek se?",
         "Mummy",
         "family",
         "Haanji Mummy, main 2:15 PM par hostel theek se pahunch gaya tha aur lunch bhi kar liya."),
        ("- Status Note: Production hotfix deployed in v1.4.2; all health checks passing.",
         "Is the production hotfix deployed?",
         "Vikram Team Lead",
         "colleague",
         "Yes Vikram, v1.4.2 has been deployed to production and all health checks are passing."),
        ("- Status Note: Aman will review open pull requests today between 3 PM and 5 PM.",
         "When can you review PR #89?",
         "Sneha Dev",
         "colleague",
         "Aman will review PR #89 today between 3:00 PM and 5:00 PM."),
    ]

    for i in range(50):
        notes, u_msg, cname, tier, asst_msg = grounded_templates[i % len(grounded_templates)]
        sys_p = build_sys_prompt(
            tier=tier,
            tone="formal_english" if tier in ("professor", "colleague") else "polite_hinglish",
            contact_name=cname,
            relation_rule=f"Recipient tier is {tier}. Adhere strictly to reference notes.",
            tone_rule="Accurate and grounded.",
            reference_notes=notes,
        )
        examples.append({"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": u_msg}, {"role": "assistant", "content": asst_msg}]})

    return examples


# ==============================================================================
# MAIN COMPOSER
# ==============================================================================
def main():
    print("=" * 80)
    print("Building EchoText Multi-Tone & Multi-Tier Fine-Tuning Dataset (1,000+ items)...")
    print("=" * 80)

    # 1. Load user's real chats
    friend_chats = load_user_friend_chats(USER_UPLOADED_PATH)
    print(f"[+] Loaded {len(friend_chats)} friend chats from user's upload.")

    # 2. Generate other tiers
    prof_chats = generate_professor_examples()
    print(f"[+] Generated {len(prof_chats)} Professor / Academic examples (Formal English & Polite Hinglish).")

    family_chats = generate_family_examples()
    print(f"[+] Generated {len(family_chats)} Family & Elders examples (Warm Hindi/Hinglish).")

    colleague_chats = generate_colleague_examples()
    print(f"[+] Generated {len(colleague_chats)} Colleague & Professional examples (Formal & Concise).")

    acq_chats = generate_acquaintance_unknown_examples()
    print(f"[+] Generated {len(acq_chats)} Acquaintance & Unknown contacts examples.")

    grounding_chats = generate_grounding_unsure_examples()
    print(f"[+] Generated {len(grounding_chats)} Strict Grounding & [UNSURE] holding examples.")

    all_data = friend_chats + prof_chats + family_chats + colleague_chats + acq_chats + grounding_chats
    random.seed(42)
    random.shuffle(all_data)

    print(f"\n[*] Total compiled dataset size: {len(all_data)} conversations.")

    # Write out JSONL
    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATASET_PATH, "w", encoding="utf-8") as f:
        for row in all_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Successfully exported dataset to: {OUTPUT_DATASET_PATH}")

    # Copy to project root for convenience
    root_out = Path(__file__).resolve().parent.parent / "echotext_1000_training_dataset.jsonl"
    try:
        with open(root_out, "w", encoding="utf-8") as f:
            for row in all_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[OK] Also mirrored dataset to project root: {root_out}")
    except Exception as e:
        print(f"Notice: Root copy skipped: {e}")

    # Validate output
    print("\n" + "=" * 80)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Rows:                   {len(all_data)}")
    print(f"Friend (Casual Hinglish):     {len(friend_chats)}")
    print(f"Professor (Academic Formal):  {len(prof_chats)}")
    print(f"Family (Warm Hindi/Hinglish): {len(family_chats)}")
    print(f"Colleague (Workplace/Concise):{len(colleague_chats)}")
    print(f"Acquaintance / Unknown:       {len(acq_chats)}")
    print(f"Grounding / [UNSURE] Holds:   {len(grounding_chats)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
