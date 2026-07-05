"""Golden-set corpus for the ``eval_recall`` harness (Memory-Roadmap §2.7).

Plain data + frozen dataclasses — no Django/Neo4j imports. Facts/entities/turns
are referenced by human-readable **keys**; the harness seeds them and builds the
``key → stored-id`` maps from the returned objects (ids are minted at store
time), then resolves golden expectations to ids for scoring.

Corpus rules:
- Every claim is textually distinct (the store's claim-hash dedup would retire
  re-seeded duplicates).
- ``expected_fact_keys[0]`` is the *best* answer (scored by MRR); the full list
  is the relevant set for recall@k.
- ``negative`` queries assert that specific near-miss distractor facts (about
  *other* people) do NOT surface in the top results for a user-scoped ask —
  an attribution/precision probe, reported as ``abstention_pass_rate``.
- ``callback`` queries score against seeded verbatim turns
  (``expected_turn_keys``) instead of facts.
- The corpus is a seam (D6): ``load_corpus("builtin")`` today; external corpora
  (LoCoMo, LongMemEval-S) become adapters returning the same ``CorpusSpec``.
"""

from dataclasses import dataclass

CATEGORIES = ("single_hop", "paraphrase", "multi_hop", "temporal", "callback", "negative")


@dataclass(frozen=True)
class SeedFact:
    key: str
    claim: str
    entity_keys: tuple[str, ...] = ()
    confidence: float = 0.9
    temporal_context: str | None = None  # None → store default ("current")


@dataclass(frozen=True)
class SeedEntity:
    key: str
    name: str
    type: str
    description: str = ""


@dataclass(frozen=True)
class SeedTurn:
    key: str
    content: str
    role: str = "user"


@dataclass(frozen=True)
class GoldenQuery:
    name: str
    category: str  # one of CATEGORIES
    query: str
    expected_fact_keys: tuple[str, ...] = ()  # [0] = best answer (MRR)
    expected_turn_keys: tuple[str, ...] = ()  # callback category
    forbid_fact_keys: tuple[str, ...] = ()  # negative category
    note: str = ""


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    facts: tuple[SeedFact, ...]
    entities: tuple[SeedEntity, ...]
    turns: tuple[SeedTurn, ...]
    queries: tuple[GoldenQuery, ...]

    def fact_keys(self) -> set[str]:
        return {f.key for f in self.facts}

    def entity_keys(self) -> set[str]:
        return {e.key for e in self.entities}

    def turn_keys(self) -> set[str]:
        return {t.key for t in self.turns}


# ---------------------------------------------------------------------------
# Builtin corpus — one coherent fictional persona so multi-hop chains and
# temporal pairs are natural. Abstract/reasoning-flavored per repo convention.
# ---------------------------------------------------------------------------

_ENTITIES = (
    SeedEntity("amara", "Amara", "person", "the user's sister, teaches philosophy"),
    SeedEntity("dr_osei", "Dr. Osei", "person", "the user's mentor, decision theorist"),
    SeedEntity("lena", "Lena", "person", "the user's climbing partner"),
    SeedEntity("helios", "Helios Institute", "organization", "the user's current employer"),
    SeedEntity("northwind", "Northwind Observatory", "organization", "the user's previous employer"),
    SeedEntity("lisbon", "Lisbon", "location", "the user's current city"),
    SeedEntity("tromso", "Tromsø", "location", "the user's previous city"),
    SeedEntity("kyoto", "Kyoto", "location", "planned November trip destination"),
    SeedEntity("aurora_atlas", "Aurora Atlas", "project", "sky-survey cataloging project the user leads"),
    SeedEntity("cartographer", "Cartographer", "project", "the user's essay series on mental maps"),
)

_FACTS = (
    # -- identity / preferences ------------------------------------------------
    SeedFact("fav_cuisine", "User's favorite cuisine is Ethiopian food."),
    SeedFact("diet", "User is vegetarian."),
    SeedFact("allergy", "User is allergic to peanuts."),
    SeedFact("tea_pref", "User drinks oolong tea while working."),
    SeedFact("wake_time", "User wakes up at 5:30 in the morning to write."),
    SeedFact("philosophy", "User practices Stoic negative visualization each morning."),
    SeedFact("book_fav", "User's favorite book is Meditations by Marcus Aurelius."),
    SeedFact("birthday_month", "User's birthday is in March."),
    SeedFact("journal", "User keeps a decision journal for major choices."),
    SeedFact("no_car", "User does not own a car and cycles everywhere."),
    SeedFact("savings_goal", "User is saving up for a telescope upgrade."),
    SeedFact("degree", "User holds a degree in atmospheric physics."),
    # -- places (temporal pair) -------------------------------------------------
    SeedFact("home_city", "User lives in Lisbon.", ("lisbon",), temporal_context="current"),
    SeedFact("home_city_past", "User previously lived in Tromsø.", ("tromso",), temporal_context="past"),
    # -- work (temporal pair) ---------------------------------------------------
    SeedFact("employer_current", "User works as a research analyst at the Helios Institute.",
             ("helios",), temporal_context="current"),
    SeedFact("employer_past", "User used to work at Northwind Observatory.",
             ("northwind",), temporal_context="past"),
    SeedFact("project_aurora", "User leads the Aurora Atlas project at the Helios Institute.",
             ("aurora_atlas", "helios")),
    SeedFact("aurora_goal", "The Aurora Atlas project catalogs auroral patterns across decades of sky surveys.",
             ("aurora_atlas",)),
    SeedFact("aurora_deadline", "The Aurora Atlas interim review is scheduled for September.",
             ("aurora_atlas",), temporal_context="future"),
    # -- people ------------------------------------------------------------------
    SeedFact("sister_amara", "User's sister is named Amara.", ("amara",)),
    SeedFact("amara_job", "Amara teaches philosophy at a university in Porto.", ("amara",)),
    SeedFact("mentor", "User's mentor is Dr. Osei.", ("dr_osei",)),
    SeedFact("osei_field", "Dr. Osei specializes in decision theory.", ("dr_osei",)),
    SeedFact("lena_friend", "User's climbing partner is Lena.", ("lena",)),
    SeedFact("lena_trait", "Lena maps cave systems as a hobby.", ("lena",)),
    # -- hobbies / activities -----------------------------------------------------
    SeedFact("chess_rating", "User's online chess rating is around 1850."),
    SeedFact("chess_opening", "User prefers playing the Caro-Kann defense in chess."),
    SeedFact("hobby_astro", "User practices astrophotography on clear weekends."),
    SeedFact("camera", "User shoots astrophotography with a Fuji X-T5 camera."),
    SeedFact("climbing_gym", "User climbs at an indoor bouldering gym on Tuesdays."),
    SeedFact("podcast", "User hosts a monthly podcast about reasoning under uncertainty."),
    SeedFact("essay_series", "User writes an essay series called Cartographer.", ("cartographer",)),
    SeedFact("essay_topic", "The Cartographer essays explore how people build mental maps of complex systems.",
             ("cartographer",)),
    # -- languages (temporal pair) / misc past -------------------------------------
    SeedFact("language_learning", "User is learning Portuguese.", temporal_context="current"),
    SeedFact("language_past", "User studied Norwegian while living in Tromsø.",
             ("tromso",), temporal_context="past"),
    SeedFact("piano", "User played piano as a child but no longer practices.",
             temporal_context="past"),
    # -- travel ---------------------------------------------------------------------
    SeedFact("kyoto_trip", "User is planning a trip to Kyoto in November.",
             ("kyoto",), temporal_context="future"),
    SeedFact("kyoto_reason", "User wants to photograph autumn maples in Kyoto.", ("kyoto",)),
    # -- distractors: near-miss facts about OTHER people (attribution probes) --------
    SeedFact("d_amara_cuisine", "Amara's favorite cuisine is Japanese food.", ("amara",)),
    SeedFact("d_amara_coffee", "Amara drinks espresso every morning.", ("amara",)),
    SeedFact("d_amara_go", "Amara plays Go rather than chess.", ("amara",)),
    SeedFact("d_amara_marathon", "Amara is training for a marathon.", ("amara",)),
    SeedFact("d_lena_city", "Lena lives in Seville.", ("lena",)),
    SeedFact("d_lena_drone", "Lena shoots landscapes with a drone.", ("lena",)),
    SeedFact("d_osei_city", "Dr. Osei is based in Accra.", ("dr_osei",)),
    SeedFact("d_helios_hq", "The Helios Institute is headquartered in Geneva.", ("helios",)),
)

_TURNS = (
    SeedTurn("t_mount_fix", "I spent the evening rebalancing my telescope mount; the counterweight kept slipping."),
    SeedTurn("t_moral_luck", "Amara and I disagreed about moral luck, but we landed on treating outcomes and intentions separately."),
    SeedTurn("t_decision_grid", "I decided to use a weighted pros-and-cons grid with a 24-hour cooling-off period for the job decision."),
    SeedTurn("t_ryokan", "I booked a small ryokan near the Philosopher's Path for the November trip."),
    SeedTurn("t_chess_blunder", "I hung my queen in yesterday's tournament game and lost in 28 moves."),
    SeedTurn("t_essay_draft", "The third Cartographer essay draft is about why subway maps distort distance but preserve order."),
    SeedTurn("t_podcast_guest", "Next month's podcast guest studies how forecasters calibrate their confidence."),
    SeedTurn("t_finger_injury", "I tweaked a finger pulley on the overhang route, so I'm taping it for two weeks."),
    SeedTurn("t_puer_gift", "Lena gifted me an aged sheng puer cake for finishing the atlas milestone."),
    SeedTurn("t_review_charts", "For the September review I'm preparing three charts: coverage, error rates, and archive growth."),
)

_QUERIES = (
    # ---- single_hop (12): near-direct asks --------------------------------------
    GoldenQuery("q_fav_cuisine", "single_hop", "What's my favorite cuisine?", ("fav_cuisine",)),
    GoldenQuery("q_sister", "single_hop", "What is my sister's name?", ("sister_amara",)),
    GoldenQuery("q_rating", "single_hop", "What's my chess rating?", ("chess_rating",)),
    GoldenQuery("q_camera", "single_hop", "Which camera do I use for astrophotography?", ("camera",)),
    GoldenQuery("q_allergy", "single_hop", "What food am I allergic to?", ("allergy",)),
    GoldenQuery("q_wake", "single_hop", "What time do I wake up?", ("wake_time",)),
    GoldenQuery("q_book", "single_hop", "What's my favorite book?", ("book_fav",)),
    GoldenQuery("q_tea", "single_hop", "What do I drink while working?", ("tea_pref",)),
    GoldenQuery("q_degree", "single_hop", "What subject is my degree in?", ("degree",)),
    GoldenQuery("q_birthday", "single_hop", "When is my birthday?", ("birthday_month",)),
    GoldenQuery("q_gym_day", "single_hop", "Which day do I go bouldering?", ("climbing_gym",)),
    GoldenQuery("q_journal", "single_hop", "What do I keep for major choices?", ("journal",)),
    # ---- paraphrase (12): same target, different surface form ---------------------
    # Some stay inside the RecallLayer expansion-synonym domains (live/work/
    # favorite/birthday), several deliberately sit OUTSIDE them so the category
    # measures embedding robustness, not the lookup table.
    GoldenQuery("q_p_residence", "paraphrase", "Which city is home for me these days?", ("home_city",)),
    GoldenQuery("q_p_job", "paraphrase", "Who employs me right now?", ("employer_current",)),
    GoldenQuery("q_p_veg", "paraphrase", "Do I eat meat?", ("diet",)),
    GoldenQuery("q_p_transport", "paraphrase", "How do I usually get around town?", ("no_car",)),
    GoldenQuery("q_p_ritual", "paraphrase", "What contemplative exercise do I do each morning?", ("philosophy",)),
    GoldenQuery("q_p_opening", "paraphrase", "Which chess opening do I favor?", ("chess_opening",)),
    GoldenQuery("q_p_show", "paraphrase", "What audio show do I run?", ("podcast",)),
    GoldenQuery("q_p_essays", "paraphrase", "What's the name of my writing project?", ("essay_series",)),
    GoldenQuery("q_p_savings", "paraphrase", "What equipment am I putting money aside for?", ("savings_goal",)),
    GoldenQuery("q_p_lang", "paraphrase", "Which language am I currently studying?", ("language_learning",)),
    GoldenQuery("q_p_night_hobby", "paraphrase", "What do I do outdoors on clear weekend nights?", ("hobby_astro",)),
    GoldenQuery("q_p_amara_work", "paraphrase", "What does Amara do for a living?", ("amara_job",)),
    # ---- multi_hop (8): answer sits one entity-link away ---------------------------
    GoldenQuery("q_m_mentor_field", "multi_hop", "What field does my mentor work in?",
                ("osei_field", "mentor"), note="user → mentor (Dr. Osei) → specialty"),
    GoldenQuery("q_m_sister_city", "multi_hop", "In which city does my sister teach?",
                ("amara_job", "sister_amara"), note="user → sister (Amara) → Porto"),
    GoldenQuery("q_m_employer_project", "multi_hop", "What project do I lead at my workplace?",
                ("project_aurora", "employer_current"), note="user → Helios → Aurora Atlas"),
    GoldenQuery("q_m_project_goal", "multi_hop", "What does the atlas project I lead actually catalog?",
                ("aurora_goal", "project_aurora"), note="user → Aurora Atlas → its goal"),
    GoldenQuery("q_m_partner_hobby", "multi_hop", "What unusual hobby does my climbing partner have?",
                ("lena_trait", "lena_friend"), note="user → Lena → cave mapping"),
    GoldenQuery("q_m_past_lang", "multi_hop", "Which language did I study when I lived up north?",
                ("language_past", "home_city_past"), note="user → Tromsø era → Norwegian"),
    GoldenQuery("q_m_trip_purpose", "multi_hop", "Why am I traveling to Japan this autumn?",
                ("kyoto_reason", "kyoto_trip"), note="user → Kyoto trip → maples"),
    GoldenQuery("q_m_essay_theme", "multi_hop", "What are my essays actually about?",
                ("essay_topic", "essay_series"), note="user → Cartographer → theme"),
    # ---- temporal (8): current-vs-past disambiguation --------------------------------
    GoldenQuery("q_t_employer_now", "temporal", "Where do I work now?", ("employer_current",),
                note="must prefer current over Northwind (past)"),
    GoldenQuery("q_t_employer_before", "temporal", "Where did I work before my current job?",
                ("employer_past",)),
    GoldenQuery("q_t_city_now", "temporal", "Where do I live currently?", ("home_city",),
                note="must prefer current over Tromsø (past)"),
    GoldenQuery("q_t_city_before", "temporal", "Where did I live before moving south?",
                ("home_city_past",)),
    GoldenQuery("q_t_piano", "temporal", "Did I ever play a musical instrument?", ("piano",)),
    GoldenQuery("q_t_review", "temporal", "What's scheduled for September?", ("aurora_deadline",)),
    GoldenQuery("q_t_trip_when", "temporal", "What trip do I have coming up in November?", ("kyoto_trip",)),
    GoldenQuery("q_t_lang_now", "temporal", "What language am I learning at the moment?",
                ("language_learning",), note="must prefer current over Norwegian (past)"),
    # ---- callback (6): answer lives in a verbatim turn ---------------------------------
    GoldenQuery("q_c_mount", "callback", "What did I do to my telescope mount recently?",
                expected_turn_keys=("t_mount_fix",)),
    GoldenQuery("q_c_moral_luck", "callback", "How did Amara and I resolve our philosophical disagreement?",
                expected_turn_keys=("t_moral_luck",)),
    GoldenQuery("q_c_ryokan", "callback", "Where am I staying in Kyoto?",
                expected_turn_keys=("t_ryokan",)),
    GoldenQuery("q_c_blunder", "callback", "How did my last tournament chess game go?",
                expected_turn_keys=("t_chess_blunder",)),
    GoldenQuery("q_c_injury", "callback", "What happened to my finger at the climbing gym?",
                expected_turn_keys=("t_finger_injury",)),
    GoldenQuery("q_c_charts", "callback", "What am I preparing for the interim review?",
                expected_turn_keys=("t_review_charts",)),
    # ---- negative (6): near-miss distractors about others must NOT surface --------------
    GoldenQuery("q_n_espresso", "negative", "Do I drink espresso every morning?",
                forbid_fact_keys=("d_amara_coffee",), note="that's Amara, not the user"),
    GoldenQuery("q_n_marathon", "negative", "Am I training for a marathon?",
                forbid_fact_keys=("d_amara_marathon",)),
    GoldenQuery("q_n_go", "negative", "Do I play Go?",
                forbid_fact_keys=("d_amara_go",)),
    GoldenQuery("q_n_drone", "negative", "Do I own a drone?",
                forbid_fact_keys=("d_lena_drone",)),
    GoldenQuery("q_n_seville", "negative", "Have I ever lived in Seville?",
                forbid_fact_keys=("d_lena_city",)),
    GoldenQuery("q_n_accra", "negative", "Am I based in Accra?",
                forbid_fact_keys=("d_osei_city",)),
)

_BUILTIN = CorpusSpec(
    name="builtin",
    facts=_FACTS,
    entities=_ENTITIES,
    turns=_TURNS,
    queries=_QUERIES,
)

_CORPORA = {"builtin": _BUILTIN}


def load_corpus(name: str = "builtin") -> CorpusSpec:
    """Return a corpus by name. External corpora (LoCoMo, LongMemEval-S) plug in
    here as adapters producing a ``CorpusSpec`` — see Memory-Roadmap §2.7."""
    try:
        return _CORPORA[name]
    except KeyError:
        raise ValueError(
            f"Unknown corpus {name!r}. Available: {sorted(_CORPORA)}"
        ) from None
