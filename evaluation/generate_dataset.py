"""Generate a balanced TokenFlow evaluation dataset (configurable size)."""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.dataset_pools_extra import (
    FACTUAL_CS_EXTRA,
    FACTUAL_GENERAL_EXTRA,
    RAG_QA_EXTRA,
    REASONING_EXTRA,
)

DEFAULT_DATASET_SIZE = 1000

# Of the factual slice at size 500 (175 items): 60 CS + 115 general; scales with --size
FACTUAL_CS_SLOTS_AT_500 = 60

# Target mix: semantic cache (cache_pair), RAG (top-k), routing (reasoning), general quality (factual)
CATEGORY_RATIOS = {
    "factual": 0.35,
    "reasoning": 0.25,
    "rag": 0.25,
    "cache_pair": 0.15,
}

FACTUAL_CS = [
    ("What is a mutex?", "A mutex is a synchronization primitive that ensures only one thread can access a shared resource at a time."),
    ("What is the difference between TCP and UDP?", "TCP is connection-oriented, reliable, and ordered. UDP is connectionless, faster, and does not guarantee delivery or order."),
    ("What is an API?", "An API is a defined interface that lets software components communicate with each other."),
    ("What is DNS?", "DNS translates human-readable domain names into IP addresses."),
    ("What is HTTPS?", "HTTPS is HTTP secured with TLS encryption and certificate-based authentication."),
    ("What is JSON?", "JSON is a lightweight text format for structured data exchange using key-value pairs and arrays."),
    ("What is caching?", "Caching stores frequently accessed data in fast storage to reduce latency and load."),
    ("What is a vector embedding?", "An embedding is a numeric vector representation of text that captures semantic meaning."),
    ("What is tokenization?", "Tokenization splits text into tokens, the units models process for billing and inference."),
    ("What is RAG?", "RAG retrieves external documents at query time to ground model answers in relevant context."),
] + FACTUAL_CS_EXTRA

FACTUAL_GENERAL = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("What gas do plants absorb from the atmosphere?", "Plants absorb carbon dioxide from the atmosphere for photosynthesis."),
    ("What is the boiling point of water at sea level in Celsius?", "Water boils at 100 degrees Celsius at standard sea-level pressure."),
    ("How many continents are there on Earth?", "Earth is commonly divided into seven continents."),
    ("What is photosynthesis?", "Photosynthesis is the process plants use to convert light energy into chemical energy."),
    ("What is the speed of light in a vacuum?", "Light travels at about 299,792 kilometers per second in a vacuum."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("In which year did the Titanic sink?", "The Titanic sank in 1912."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("What is the chemical symbol for water?", "The chemical formula for water is H2O."),
    ("What organ pumps blood through the human body?", "The heart pumps blood through the human body."),
    ("What vitamin is produced when skin is exposed to sunlight?", "Sunlight exposure helps the skin produce vitamin D."),
    ("What is the main language spoken in Brazil?", "Portuguese is the main official language of Brazil."),
    ("What currency is used in the United Kingdom?", "The United Kingdom uses the pound sterling."),
    ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain above sea level."),
    ("What ocean lies between the Americas and Europe?", "The Atlantic Ocean lies between the Americas and Europe."),
    ("What is democracy?", "Democracy is a system of government where power is vested in the people, often through voting."),
    ("What is inflation?", "Inflation is a sustained increase in the general price level of goods and services."),
    ("What is a mortgage?", "A mortgage is a loan used to purchase real estate, secured by the property."),
    ("What does UNESCO stand for?", "UNESCO stands for the United Nations Educational, Scientific and Cultural Organization."),
    ("What is the Pythagorean theorem?", "In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides."),
    ("What is pi approximately equal to?", "Pi is approximately 3.14159."),
    ("What is mitosis?", "Mitosis is cell division that produces two genetically identical daughter cells."),
    ("What is gravity?", "Gravity is the attractive force between masses."),
    ("What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius at standard pressure."),
    ("What is the primary source of energy for Earth's climate system?", "The Sun is the primary source of energy for Earth's climate system."),
    ("What is Buddhism?", "Buddhism is a religion and philosophy founded on teachings attributed to Siddhartha Gautama."),
    ("What is the Sahara?", "The Sahara is a large hot desert in North Africa."),
    ("What is the Amazon River known for?", "The Amazon River is one of the largest rivers by discharge and drains much of South America."),
    ("What is jazz?", "Jazz is a music genre originating in African American communities, known for improvisation and swing."),
    ("What is olive oil made from?", "Olive oil is made by pressing olives."),
    ("What is gluten?", "Gluten is a group of proteins found in wheat and related grains."),
    ("What is a passport?", "A passport is an official travel document identifying nationality for international travel."),
    ("What is renewable energy?", "Renewable energy comes from sources that replenish naturally, such as solar and wind."),
    ("What is the Nobel Prize?", "The Nobel Prize recognizes outstanding contributions in fields such as physics, chemistry, medicine, literature, and peace."),
    ("What is the Great Wall of China?", "The Great Wall of China is a historic series of fortifications built to protect Chinese states."),
] + FACTUAL_GENERAL_EXTRA

FACTUAL = FACTUAL_CS + FACTUAL_GENERAL

REASONING = [
    ("Why might a web app feel slow even when CPU usage is low?", "Bottlenecks may be I/O, network latency, database queries, lock contention, or waiting on external APIs rather than CPU."),
    ("When should you use a queue instead of a stack?", "Use a queue for FIFO processing like job scheduling, task pipelines, or breadth-first traversal."),
    ("Why is idempotency important in distributed systems?", "Retries are common in distributed systems; idempotent operations prevent duplicate side effects when requests are replayed."),
    ("How does a cache reduce cost for LLM applications?", "Cache hits avoid repeated model calls and token processing for similar requests."),
    ("Why might two semantically similar questions still need different answers?", "Context, intent, constraints, or domain specificity can change the correct answer even when wording is similar."),
    ("When is a smaller LLM sufficient?", "Simple factual questions, short prompts, and low reasoning depth often work well with smaller models."),
    ("Why can high cache hit rate still be bad?", "Incorrect cache hits return wrong answers, harming quality despite saving cost."),
    ("How does chunk ranking help RAG?", "It sends only the most relevant context, reducing tokens while preserving answer quality."),
    ("When does routing to a large model help most?", "Complex reasoning, multi-step analysis, debugging, and nuanced comparisons usually need stronger models."),
    ("Why should you tune cache thresholds on a dev set?", "It prevents overfitting threshold choices to the same data used for final reporting."),
    ("Why might rain occur on one side of a mountain but not the other?", "Orographic lift can cool moist air and cause precipitation on the windward side while the leeward side stays drier."),
    ("Why do airlines overbook flights?", "Airlines overbook to offset expected no-shows while still filling seats, accepting a small risk of denied boarding."),
    ("How would you choose between train and car for a trip?", "Consider distance, cost, time, convenience, luggage, and whether you need flexibility at the destination."),
    ("Why does cutting interest rates sometimes stimulate spending?", "Lower rates reduce borrowing costs and can encourage investment and consumer spending."),
    ("Why is it risky to share passwords across websites?", "If one site is breached, reused passwords let attackers access your other accounts."),
    ("Why do vaccines reduce disease spread?", "Vaccines train the immune system and lower infection rates, reducing transmission in the population."),
    ("Why might two countries trade even when one is more efficient at everything?", "Comparative advantage means both can gain by specializing where relative efficiency is highest."),
    ("How do you decide whether to rent or buy a home?", "Compare monthly costs, how long you will stay, maintenance, flexibility, and local market conditions."),
    ("Why can identical recipes taste different at altitude?", "Lower air pressure changes boiling points and evaporation, affecting baking and cooking times."),
    ("Why is sample size important in surveys?", "Larger, representative samples reduce random error and make estimates more reliable."),
] + REASONING_EXTRA

# Unique RAG scenarios (20 chunks each, one relevant index) — not cloned variants
RAG_SCENARIOS = [
    {
        "question": "According to the docs, what port does TokenFlow use by default?",
        "chunks": [
            "TokenFlow runs on port 8000 by default when started with uvicorn.",
            "PostgreSQL in docker-compose exposes port 5432.",
            "The dashboard reads evaluation/results/latest.json.",
            "Cache entries store question, answer, and embedding vectors.",
            "Gemini models are configured via environment variables.",
            "The health endpoint returns status ok.",
            "Benchmark results include tokens, latency, and quality scores.",
            "RAG selection keeps the top K chunks by embedding similarity.",
            "Model routing uses complexity heuristics based on prompt length.",
            "The dev/test split uses an 80/20 holdout from the evaluation dataset.",
            "Semantic cache uses pgvector cosine distance for lookup.",
            "Feature flags enable cache, rag selection, and routing independently.",
            "The judge rubric scores correctness, completeness, and relevance.",
            "Requests are logged to evaluation/logs/requests.jsonl.",
            "Docker compose mounts database/init.sql on startup.",
            "Cache threshold tuning should use the dev set only.",
            "Combined mode enables all optimizations together.",
            "Baseline chat endpoint disables all optimizations.",
            "Small models handle simple prompts in routing.",
            "Top K for RAG defaults to five chunks.",
        ],
        "relevant": [0],
        "answer": "TokenFlow uses port 8000 by default.",
    },
    {
        "question": "Which database extension is required for semantic cache?",
        "chunks": [
            "TokenFlow stores embeddings in PostgreSQL using pgvector.",
            "Redis can be added later for hot cache layers.",
            "SQLite is not used in the default setup.",
            "The cache table includes question, answer, embedding, created_at.",
            "Embeddings use Gemini embedding models.",
            "Connection string is configured with DATABASE_URL.",
            "init.sql creates the cache_entries table.",
            "Benchmark runner supports multiple optimization modes.",
            "Judge evaluation uses a 0-4 correctness rubric.",
            "Threshold sweep tests 0.80, 0.90, 0.95, 0.98.",
            "Report generator outputs markdown and CSV.",
            "Dashboard uses Chart.js for bar charts.",
            "Pipeline order is cache, rag, router, then llm.",
            "Optimized endpoint reads feature flags from env.",
            "Cosine similarity compares chunk relevance.",
            "Complex keywords increase routing score.",
            "Held-out test set is 20% of evaluation questions.",
            "Manual validation samples random dev questions.",
            "pytest covers router, rag, and cache logic.",
            "IVFFlat or no index may be used for high-dimensional vectors.",
        ],
        "relevant": [0],
        "answer": "The pgvector extension is required for semantic cache storage and similarity search.",
    },
    {
        "question": "What is the company remote-work policy on core hours?",
        "chunks": [
            "Core collaboration hours are 10:00 AM to 3:00 PM in your local time zone.",
            "The cafeteria serves lunch from 11:30 to 1:30.",
            "Parking permits renew every January.",
            "Employees may work remotely up to three days per week with manager approval.",
            "The holiday party is scheduled for December.",
            "Health insurance open enrollment is in November.",
            "Laptops must use full-disk encryption.",
            "Travel bookings require 14-day advance notice for international trips.",
            "The dress code is business casual.",
            "Volunteer day grants eight paid hours annually.",
            "Meeting rooms are booked via the internal calendar.",
            "Passwords must be at least 12 characters.",
            "The office closes on national public holidays.",
            "Expense reports are due within 30 days.",
            "Training budget is $1,000 per employee per year.",
            "Customer support SLA is 24 hours for email.",
            "Brand colors are defined in the style guide.",
            "Slack is the default chat tool.",
            "GitHub is used for source control.",
            "Performance reviews occur twice per year.",
        ],
        "relevant": [0],
        "answer": "Core collaboration hours are 10:00 AM to 3:00 PM in your local time zone.",
    },
    {
        "question": "How many paid vacation days do new hires receive?",
        "chunks": [
            "New hires receive 15 paid vacation days per year, prorated in the first year.",
            "Sick leave is separate and unlimited with manager notification.",
            "The office building has 12 floors.",
            "401k matching is 4% after 90 days.",
            "Lunch stipends were discontinued in 2019.",
            "Conference attendance requires VP approval.",
            "Interns are not eligible for vacation accrual.",
            "Parental leave policy is 12 weeks primary caregiver.",
            "Commuter benefits cover public transit up to $300 monthly.",
            "Stock options vest over four years.",
            "The CEO town hall is monthly.",
            "Desk hoteling applies on Fridays.",
            "Security badges must be visible on campus.",
            "Fire drills occur quarterly.",
            "The wiki is hosted on Confluence.",
            "Customer NPS target is 45.",
            "Data retention for logs is 90 days.",
            "VPN is required off-site.",
            "Marketing launches need legal review.",
            "Office wifi SSID is CorpSecure.",
        ],
        "relevant": [0],
        "answer": "New hires receive 15 paid vacation days per year, prorated in the first year.",
    },
    {
        "question": "What temperature should the chicken reach for safe consumption?",
        "chunks": [
            "Cook poultry to an internal temperature of 165°F (74°C).",
            "Preheat the oven to 350°F for most casseroles.",
            "Store leftovers within two hours of cooking.",
            "Marinate fish for no more than 30 minutes with citrus.",
            "Brown rice takes about 45 minutes to simmer.",
            "Knives should be sharpened monthly.",
            "Basil bruises easily; tear instead of chop.",
            "Olive oil smoke point varies by grade.",
            "Yeast proofs best around 110°F.",
            "Shellfish allergies require separate prep surfaces.",
            "Cast iron should be dried immediately after washing.",
            "Quinoa rinsing removes bitter saponins.",
            "Garlic burns quickly over high heat.",
            "Rest steaks five minutes before slicing.",
            "Freezer temperature should be 0°F or below.",
            "Acidic tomatoes can react with aluminum pans.",
            "Mise en place reduces cooking errors.",
            "Honey should not be given to infants under one year.",
            "Cross-contamination is a leading cause of foodborne illness.",
            "Vegetable stock simmers for 45–60 minutes.",
        ],
        "relevant": [0],
        "answer": "Cook poultry to an internal temperature of 165°F (74°C).",
    },
    {
        "question": "Which museum pass includes free entry on the first Sunday?",
        "chunks": [
            "The City Arts Pass includes free entry to participating museums on the first Sunday of each month.",
            "Metro lines 2 and 5 stop near the riverfront.",
            "Street parking is metered until 8 PM weekdays.",
            "The botanical garden opens at 9 AM year-round.",
            "Bike share stations accept contactless payment.",
            "The summer festival runs July 15–30.",
            "Library cards are free for residents.",
            "Tour buses depart from Central Plaza.",
            "The aquarium closes on Mondays for maintenance.",
            "Historic district walking tours start at 10 AM.",
            "Snow routes affect bus schedules in winter.",
            "The observatory requires advance tickets on weekends.",
            "River cruises last 90 minutes.",
            "Hostels require passport at check-in.",
            "Tap water is safe to drink citywide.",
            "Emergency number is 112.",
            "Airport shuttle runs every 30 minutes.",
            "Sales tax on meals is 8%.",
            "Smoking is banned in indoor public places.",
            "The night market operates Fridays only.",
        ],
        "relevant": [0],
        "answer": "The City Arts Pass includes free entry to participating museums on the first Sunday of each month.",
    },
    {
        "question": "What is the warranty period for the Model X battery?",
        "chunks": [
            "Model X batteries are covered by an 8-year or 100,000-mile warranty, whichever comes first.",
            "Tire rotation is recommended every 7,500 miles.",
            "The infotainment system supports Apple CarPlay.",
            "Roof racks have a 165-pound limit.",
            "Child seats must use LATCH anchors.",
            "Headlight washers activate with windshield fluid.",
            "The spare tire kit is optional on base trim.",
            "Software updates install over Wi-Fi.",
            "Premium sound has 12 speakers.",
            "Tow mode disables parking sensors.",
            "Floor mats are all-weather standard on outdoor package.",
            "Paint touch-up codes are on the door jamb sticker.",
            "Adaptive cruise requires radar calibration after front-end work.",
            "The owner's manual is available as PDF.",
            "Dealership service hours are 7 AM–6 PM weekdays.",
            "Roadside assistance covers four years.",
            "Winter tires are recommended below 40°F for performance trims.",
            "Key fob battery is CR2032.",
            "Cargo cover is removable.",
            "VIN is visible through the windshield.",
        ],
        "relevant": [0],
        "answer": "Model X batteries are covered by an 8-year or 100,000-mile warranty, whichever comes first.",
    },
    {
        "question": "When is the assignment due for History 101?",
        "chunks": [
            "History 101 Essay 2 is due on October 12 at 11:59 PM on the course portal.",
            "Office hours are Tuesdays 2–4 PM in Hall B.",
            "Midterm covers chapters 4–7.",
            "Required text is Thompson, Modern Europe.",
            "Discussion posts are weekly on Thursdays.",
            "Plagiarism policy follows university honor code.",
            "Late work loses 10% per day up to three days.",
            "Film screening is optional extra credit.",
            "Study group meets in library room 204.",
            "Graduate TAs grade quizzes.",
            "Citation style is Chicago footnotes.",
            "Accessibility accommodations contact Disability Services.",
            "Campus Wi-Fi requires student login.",
            "Final exam is cumulative.",
            "Primary source workshop is September 20.",
            "Course syllabus lists learning outcomes.",
            "Recording lectures requires instructor consent.",
            "Bibliography must include at least five scholarly sources.",
            "Peer review draft due October 5.",
            "Library reserves hold two copies of the textbook.",
        ],
        "relevant": [0],
        "answer": "History 101 Essay 2 is due on October 12 at 11:59 PM on the course portal.",
    },
]

RAG_FILLER_CHUNKS = [
    "See appendix B for definitions.",
    "Contact the help desk for account issues.",
    "Figures are unaudited unless marked otherwise.",
    "This policy may change without notice.",
    "Archived versions are kept for seven years.",
    "Regional rules may override global defaults.",
    "Training modules cover safety basics.",
    "Mobile app notifications are opt-in.",
    "Third-party integrations require admin approval.",
    "Support hours are 9 AM–5 PM weekdays.",
    "Discounts cannot be combined unless stated.",
    "Measurements use metric units unless noted.",
    "Accessibility requests are handled within 48 hours.",
    "Marketing emails include an unsubscribe link.",
    "Inventory counts refresh nightly.",
    "Guest Wi-Fi is rate-limited.",
    "Legal review is required for external publishing.",
    "Backup generators are tested monthly.",
    "Personal data is processed under the privacy policy.",
]

# (question, single authoritative sentence used as chunk 0 and as reference answer)
RAG_QA_LINES = [
    ("What is the standard shipping time for domestic orders?", "Domestic standard shipping delivers in 5–7 business days."),
    ("How do I reset my portal password?", "Use the Forgot Password link to receive a reset email valid for one hour."),
    ("What is the daily withdrawal limit at ATMs?", "Daily ATM withdrawal limit is $500 per card."),
    ("When does the library close on Fridays?", "The main library closes at 8 PM on Fridays."),
    ("What age is required to rent a car?", "Renters must be at least 21 years old with a valid license."),
    ("Is there a fee for early ticket cancellation?", "Cancellations more than 48 hours before showtime receive a full refund minus a $5 fee."),
    ("What is the recommended daily water intake for adults?", "Many guidelines suggest about 2–3 liters of fluids per day for adults, adjusted for activity and climate."),
    ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet due to iron oxide on its surface."),
    ("What year did the first Moon landing occur?", "Apollo 11 landed on the Moon in 1969."),
    ("What is the emergency number in the European Union?", "112 is the common emergency number in the EU."),
    ("How long should you wash hands with soap?", "Wash hands with soap for at least 20 seconds."),
    ("What is the maximum carry-on size for this airline?", "Carry-on bags must not exceed 22 x 14 x 9 inches including handles."),
    ("When is property tax due in this county?", "Property tax installments are due April 30 and October 31."),
    ("What GPA is required for dean's list?", "Dean's list requires a term GPA of at least 3.5 with 12 graded credits."),
    ("What is the calorie count listed for one serving of oatmeal?", "One serving of plain oatmeal is listed as 150 calories."),
    ("Which cable is used for fast phone charging in the box?", "USB-C to USB-C cable supports fast charging for included phones."),
    ("What is the reentry period after quitting for rehire?", "Former employees may reapply after 90 days from last day worked."),
    ("How many guests may attend the wedding reception per invite?", "Each invitation admits two named guests."),
    ("What is the speed limit in the campus zone?", "Campus zone speed limit is 25 mph when children are present."),
    ("When does daylight saving time start in the US?", "Daylight saving time in the US begins on the second Sunday in March."),
    ("What is the deductible on the standard health plan?", "The standard plan has a $1,500 individual annual deductible."),
    ("Which ingredient should celiac customers avoid?", "Products containing wheat, barley, or rye gluten must be avoided."),
    ("What is the late fee for overdue books?", "Overdue books accrue $0.25 per day up to $10 per item."),
    ("How many minutes is the standard appointment slot?", "Standard primary care appointments are 20 minutes."),
    ("What is the minimum age for a senior ticket?", "Senior tickets apply to guests age 65 and older."),
    ("What is the return shipping label expiration?", "Prepaid return labels expire 14 days after issue."),
    ("Which feature flag enables semantic caching in TokenFlow?", "ENABLE_CACHE=true enables semantic caching."),
    ("Where are benchmark JSON results written?", "Benchmark output is stored under evaluation/results/ with a run id filename."),
    ("What default top-K does RAG selection use?", "RAG top-K defaults to five chunks unless configured otherwise."),
    ("What split ratio separates dev and test evaluation sets?", "Evaluation uses an 80% dev and 20% test split by question id."),
    ("What is the zoo's last entry time?", "Last entry is one hour before closing each day."),
    ("What is the recommended tire pressure for the sedan?", "Door-jamb sticker recommends 32 PSI cold for front and rear."),
    ("How many sick days do tenured teachers receive?", "Tenured teachers receive 10 paid sick days per school year."),
    ("What is the conference registration deadline?", "Early registration ends August 1; standard rates apply after."),
    ("Which file format is required for invoice uploads?", "Invoices must be uploaded as PDF under 10 MB."),
    ("What is the pet policy for this apartment lease?", "One cat or dog under 40 lbs is allowed with a $300 deposit."),
    ("When does the farmers market open?", "The farmers market opens Saturdays at 8 AM from May through October."),
    ("What is the grace period for monthly rent?", "Rent has a five-day grace period before late fees apply."),
    ("What is the student ID replacement fee?", "Replacement student IDs cost $15 at the card office."),
    ("Which ocean is the Bermuda Triangle associated with?", "The Bermuda Triangle is a region in the western North Atlantic Ocean."),
    ("What is the recommended screen break interval?", "Take a 5-minute break every hour of continuous screen use."),
    ("How many players are on the field per soccer team?", "Each soccer team fields 11 players including the goalkeeper."),
    ("What is the alcohol serving cutoff time?", "Alcohol service stops at 2 AM on licensed premises."),
    ("What is the forecasted high temperature for tomorrow?", "Tomorrow's forecast high is 72°F with partly cloudy skies."),
    ("Which vitamin is linked to citrus fruits?", "Citrus fruits are a well-known source of vitamin C."),
    ("What is the museum's free admission day?", "General admission is free on the first Tuesday of each month."),
    ("How many weeks of parental leave are paid?", "Paid parental leave is eight weeks for eligible full-time employees."),
    ("What is the minimum payment on the credit card statement?", "Minimum payment is 2% of balance or $25, whichever is greater."),
    ("When must expense receipts be submitted?", "Receipts must be submitted within 30 days of purchase."),
    ("What is the building's maximum occupancy?", "Posted maximum occupancy is 450 persons for the main hall."),
    ("Which language is taught in the beginner course?", "Beginner course teaches conversational Spanish."),
    ("What is the shelf life of opened milk in the fridge?", "Opened milk is best within 5–7 days when refrigerated at 40°F or below."),
    ("How many lanes are open for lap swim mornings?", "Lap swim has six lanes available from 6–8 AM weekdays."),
    ("What is the penalty for parking without a permit?", "Unpermitted parking fines start at $45 per violation."),
    ("What is the recommended first step for a minor kitchen burn?", "Cool the burn under cool running water for 10–20 minutes."),
    ("Which month is fiscal year-end?", "Fiscal year-end is March 31."),
    ("What is the group discount threshold?", "Groups of 15 or more receive a 10% ticket discount."),
    ("How long is the free trial for the software?", "The free trial lasts 14 days with full feature access."),
    ("What is the maximum file size for email attachments?", "Attachments over 25 MB must use the secure file link instead."),
    ("When does the pool close for winter?", "The outdoor pool closes September 15 for winterization."),
    ("What is the standard tip percentage suggested on receipts?", "Suggested gratuity on receipts is 18%, 20%, or 22%."),
    ("Which blood type is the universal donor for red cells?", "Type O negative is the universal donor for red blood cells."),
    ("What is the check-in time for hotel reservations?", "Standard check-in begins at 3 PM local property time."),
    ("How many credits are required to graduate?", "The program requires 120 credits to graduate."),
    ("What is the noise quiet hours policy?", "Quiet hours are 10 PM to 8 AM in residential buildings."),
    ("Which button submits the job application?", "Click Apply Now at the bottom of the posting to submit."),
    ("What is the recommended age for a first dental visit?", "First dental visit is recommended by age one or within six months of first tooth."),
    ("How many bags are included in basic economy?", "Basic economy includes one personal item; carry-on costs extra."),
    ("What is the solar panel warranty length?", "Solar panels carry a 25-year performance warranty."),
    ("When is the town hall meeting?", "Town hall is scheduled for November 3 at 7 PM in the community center."),
    ("What is the cookie consent retention period?", "Analytics cookies expire after 13 months unless renewed."),
    ("Which port does the API gateway listen on in staging?", "Staging API gateway listens on port 8443 with TLS."),
    ("What is the SLA for critical production incidents?", "Critical incidents have a 15-minute initial response SLA."),
    ("How many questions are in the balanced evaluation mix?", "The evaluation dataset mixes factual, reasoning, RAG, and cache-pair categories."),
    ("What similarity thresholds does cache audit test?", "Cache audit sweeps thresholds 0.80, 0.90, 0.95, and 0.98."),
    ("What does combined benchmark mode enable?", "Combined mode enables cache, RAG chunk selection, and model routing together."),
] + RAG_QA_EXTRA

CACHE_PARAPHRASES = [
    "Can you explain: {q}",
    "In simple terms, {q}",
    "Please clarify — {q}",
    "I'd like to know: {q}",
    "Could you help me understand {q}",
]

# Wording variants for scaling pools — not used for cache_pair (those use CACHE_PARAPHRASES).
NON_CACHE_VARIANTS = [
    "{q}",
    "Briefly answer: {q}",
    "In your own words, {q}",
    "For study purposes: {q}",
    "Core knowledge check — {q}",
    "Explain clearly: {q}",
    "One-paragraph answer to: {q}",
    "Key fact: what is {base}?",
    "Background topic: {q}",
    "Concept review: {q}",
    "Short explanation needed: {q}",
    "Define and illustrate: {q}",
]


def allocate_counts(target_size: int) -> dict[str, int]:
    counts = {cat: int(target_size * ratio) for cat, ratio in CATEGORY_RATIOS.items()}
    assigned = sum(counts.values())
    if assigned < target_size:
        counts["factual"] += target_size - assigned
    elif assigned > target_size:
        counts["factual"] -= assigned - target_size
    return counts


def build_extended_rag_scenarios() -> list[dict]:
    scenarios = list(RAG_SCENARIOS)
    for question, answer in RAG_QA_LINES:
        chunks = [answer] + RAG_FILLER_CHUNKS[:19]
        scenarios.append(
            {
                "question": question,
                "chunks": chunks,
                "relevant": [0],
                "answer": answer,
            }
        )
    return scenarios


ALL_RAG_SCENARIOS = build_extended_rag_scenarios()


def _base_phrase(question: str) -> str:
    return question.rstrip("?").strip()


def next_unique_question(canonical_q: str, seen: set[str]) -> str:
    """Assign a unique question string; never uses cache-paraphrase templates."""
    base = _base_phrase(canonical_q)
    q = f"{base}?"
    for variant_idx in range(500):
        for tmpl in NON_CACHE_VARIANTS:
            if "{base}" in tmpl:
                candidate = tmpl.format(base=base)
            else:
                candidate = tmpl.format(q=q)
            if not candidate.endswith("?"):
                candidate = f"{candidate}?"
            if variant_idx > 0:
                candidate = f"{candidate.rstrip('?')} (variant {variant_idx + 1})?"
            if candidate not in seen:
                return candidate
    raise RuntimeError(f"Could not allocate unique question for: {canonical_q!r}")


def expand_qa_pool(
    pool: list[tuple[str, str]], count: int, seen: set[str]
) -> list[tuple[str, str]]:
    """Return count pairs; unique question strings; prefer unused reference answers."""
    if count <= 0:
        return []
    if not pool:
        raise ValueError("expand_qa_pool requires a non-empty pool")
    ref_use: Counter[str] = Counter()
    out: list[tuple[str, str]] = []
    cursor = 0
    while len(out) < count:
        best_idx = min(
            range(len(pool)),
            key=lambda j: (ref_use[pool[j][1]], (cursor + j) % len(pool)),
        )
        q, ref = pool[best_idx]
        ref_use[ref] += 1
        cursor = (best_idx + 1) % len(pool)
        candidate = next_unique_question(q, seen)
        seen.add(candidate)
        out.append((candidate, ref))
    return out


def unique_cache_paraphrase(base_q: str, template: str, seen: set[str], pair_index: int) -> str:
    """Paraphrase for cache_pair — similar to base, unique string, uses cache templates."""
    core = base_q.rstrip("?") + "?"
    candidate = template.format(q=core)
    if candidate in seen:
        candidate = template.format(q=core.rstrip("?") + f" (follow-up {pair_index + 1})?")
    suffix = 0
    while candidate in seen:
        suffix += 1
        candidate = f"{template.format(q=core).rstrip('?')} (alt {suffix})?"
    return candidate


def make_item(
    item_id: int,
    question: str,
    reference_answer: str,
    category: str,
    context_chunks: list | None = None,
    relevant_chunk_ids: list | None = None,
    similar_to_id: int | None = None,
) -> dict:
    return {
        "id": item_id,
        "question": question,
        "reference_answer": reference_answer,
        "category": category,
        "context_chunks": context_chunks or [],
        "relevant_chunk_ids": relevant_chunk_ids or [],
        "similar_to_id": similar_to_id,
    }


def factual_cs_slot_count(factual_total: int) -> int:
    """CS factual count scales with dataset size; 60 when factual_total is 175 (size 500)."""
    if factual_total <= 0:
        return 0
    factual_at_500 = int(500 * CATEGORY_RATIOS["factual"])
    ratio = FACTUAL_CS_SLOTS_AT_500 / factual_at_500
    cs = round(factual_total * ratio)
    return min(factual_total, max(1, cs))


def build_dataset(target_size: int = DEFAULT_DATASET_SIZE) -> list[dict]:
    target_size = max(target_size, 20)
    counts = allocate_counts(target_size)
    items: list[dict] = []
    item_id = 1
    used_questions: set[str] = set()

    cs_n = factual_cs_slot_count(counts["factual"])
    gen_n = counts["factual"] - cs_n
    factual_rows = expand_qa_pool(FACTUAL_CS, cs_n, used_questions) + expand_qa_pool(
        FACTUAL_GENERAL, gen_n, used_questions
    )
    factual_ids: list[int] = []
    for q, ref in factual_rows:
        items.append(make_item(item_id, q, ref, "factual"))
        factual_ids.append(item_id)
        item_id += 1

    for q, ref in expand_qa_pool(REASONING, counts["reasoning"], used_questions):
        items.append(make_item(item_id, q, ref, "reasoning"))
        item_id += 1

    rag_pool = [(r["question"], r["answer"]) for r in ALL_RAG_SCENARIOS]
    rag_by_answer = {r["answer"]: r for r in ALL_RAG_SCENARIOS}
    for q, ref in expand_qa_pool(rag_pool, counts["rag"], used_questions):
        scenario = rag_by_answer[ref]
        items.append(
            make_item(
                item_id,
                q,
                ref,
                "rag",
                context_chunks=scenario["chunks"],
                relevant_chunk_ids=scenario["relevant"],
            )
        )
        item_id += 1

    for i in range(counts["cache_pair"]):
        base_idx = i % len(factual_ids)
        base_id = factual_ids[base_idx]
        base_q, base_ref = factual_rows[base_idx]
        template = CACHE_PARAPHRASES[i % len(CACHE_PARAPHRASES)]
        paraphrase = unique_cache_paraphrase(base_q, template, used_questions, i)
        used_questions.add(paraphrase)
        items.append(
            make_item(
                item_id,
                paraphrase,
                base_ref,
                "cache_pair",
                similar_to_id=base_id,
            )
        )
        item_id += 1

    return items[:target_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced TokenFlow evaluation dataset")
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_DATASET_SIZE,
        help=f"Total questions (default {DEFAULT_DATASET_SIZE})",
    )
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help="Regenerate evaluation/splits.json (80%% dev / 20%% test)",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.size)
    path = Path("evaluation/dataset.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    from collections import Counter

    breakdown = Counter(x["category"] for x in dataset)
    factual_n = breakdown.get("factual", 0)
    cs_n = factual_cs_slot_count(factual_n) if factual_n else 0
    print(f"Wrote {len(dataset)} questions to {path}")
    print("Category mix:", dict(breakdown))
    if factual_n:
        print(f"Factual breakdown: {cs_n} CS + {factual_n - cs_n} general")

    if args.write_splits:
        from evaluation.splits import create_splits, save_splits

        splits = create_splits(dataset)
        save_splits(splits)
        print(
            f"Wrote splits: {len(splits['dev_ids'])} dev, "
            f"{len(splits['test_ids'])} test (seed={splits['seed']})"
        )


if __name__ == "__main__":
    main()
