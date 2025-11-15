import streamlit as st
import pandas as pd
import numpy as np
import random
import io

from dataclasses import dataclass
from typing import Dict, Tuple, List

# Google Sheets imports
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe

# -------------------------------
# Streamlit config
# -------------------------------
st.set_page_config(page_title="HELMETS – League Manager", layout="wide")

# -------------------------------
# Google Sheets helpers
# -------------------------------

def get_gs_client():
    """Build a Google Sheets client from Streamlit secrets."""
    if "gcp_service_account" not in st.secrets or "SHEET_ID" not in st.secrets:
        raise RuntimeError("Google Sheets secrets not configured.")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    return gspread.authorize(creds)

@st.cache_data(ttl=60)
def read_sheet(sheet_name: str) -> pd.DataFrame:
    gc = get_gs_client()
    sh = gc.open_by_key(st.secrets["SHEET_ID"])
    ws = sh.worksheet(sheet_name)
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

def upsert_sheet(sheet_name: str, df: pd.DataFrame):
    gc = get_gs_client()
    sh = gc.open_by_key(st.secrets["SHEET_ID"])
    try:
        ws = sh.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        rows = max(100, len(df) + 20)
        cols = max(20, len(df.columns) + 5)
        ws = sh.add_worksheet(title=sheet_name, rows=str(rows), cols=str(cols))
    ws.clear()
    set_with_dataframe(ws, df, include_index=False, include_column_header=True)

# -------------------------------
# Data structures + helpers
# -------------------------------

DIVS = ["East", "North", "South", "West"]
HOME_FIELD_BONUS = 3.0

@dataclass
class Profile:
    profile_id: int
    profile_name: str
    overall: int
    offense: int
    defense: int
    qb: int
    momentum: int
    injury: int
    description: str

def make_demo_teams() -> pd.DataFrame:
    EMOJIS = ["🏈","🦅","🐻","🐯","🦬","🦁","🐺","🐻‍❄️","🐬","🦈","🔥","⚡","🌪️","🌊","❄️","🚀"]
    teams = []
    confs = ["Alpha", "Beta"]
    tid = 1
    for conf in confs:
        for d in DIVS:
            for i in range(4):  # 4 teams per division (32 teams total)
                name = f"{conf} {d} {chr(65+i)}"
                emoji = EMOJIS[(tid-1) % len(EMOJIS)]
                teams.append([tid, name, emoji, conf, d])
                tid += 1
    return pd.DataFrame(teams, columns=["team_id","team_name","emoji","conference","division"])

def make_demo_profiles(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for pid in range(1, n+1):
        overall = int(rng.integers(60, 93))
        offense = int(np.clip(rng.normal(overall, 8), 50, 99))
        defense = int(np.clip(rng.normal(overall, 8), 50, 99))
        qb      = int(np.clip(rng.normal((overall-60)/(99-60)*28 + 4, 6), 0, 32))
        momentum= int(rng.integers(0, 9))
        injury  = int(rng.integers(0, 6))
        desc = "Auto-generated profile"
        rows.append([pid, f"Profile {pid}", overall, offense, defense, qb, momentum, injury, desc])
    return pd.DataFrame(rows, columns=[
        "profile_id","profile_name","overall","offense","defense","qb","momentum","injury","description"
    ])

# -------------------------------
# Schedule + sim logic
# -------------------------------

def schedule_for_team(team_id: int, teams: pd.DataFrame, cross_conf_map: Dict[str,str]) -> List[Tuple[int,int]]:
    """Return list of (home_id, away_id) opponents for a single team."""
    row = teams.loc[teams.team_id == team_id].iloc[0]
    conf, div = row.conference, row.division

    intra_conf = teams[(teams.conference == conf) & (teams.division != div)]
    intra_ids = intra_conf.team_id.tolist()  # for 32 teams: 12 opponents

    cross_div = cross_conf_map.get(div, DIVS[0])
    other_conf = [c for c in teams.conference.unique() if c != conf]
    cross_conf = other_conf[0] if other_conf else conf
    cc_div_teams = teams[(teams.conference == cross_conf) & (teams.division == cross_div)]
    cc_div_ids = cc_div_teams.team_id.tolist()

    others = teams[(teams.conference == cross_conf) & (teams.division != cross_div)]
    others_ids = others.team_id.tolist()
    rnd = random.choice(others_ids) if others_ids else team_id

    opponents = intra_ids + cc_div_ids + [rnd]  # may not be exactly 17 for non-32-team setups
    pairs = []
    flip = True
    for opp in opponents:
        if flip:
            pairs.append((team_id, opp))
        else:
            pairs.append((opp, team_id))
        flip = not flip
    return pairs

def generate_schedule(teams: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    cross_conf_map = {d:d for d in DIVS}
    desired = {tid: schedule_for_team(tid, teams, cross_conf_map) for tid in teams.team_id}

    seen_games = set()
    all_games = []
    for tid, pairs in desired.items():
        for home, away in pairs:
            key = (home, away)
            rkey = (away, home)
            if key in seen_games or rkey in seen_games:
                continue
            seen_games.add(key)
            all_games.append(key)

    random.shuffle(all_games)
    weeks = []
    n_weeks = 17
    per_week = max(1, len(all_games) // n_weeks)
    extra = len(all_games) % n_weeks
    idx = 0
    for w in range(1, n_weeks+1):
        count = per_week + (1 if w <= extra else 0)
        for _ in range(count):
            if idx < len(all_games):
                h, a = all_games[idx]
                weeks.append((w, h, a))
                idx += 1

    sched = pd.DataFrame(weeks, columns=["week","home_team_id","away_team_id"])
    return sched

def clamp(x, lo=10, hi=90):
    return max(lo, min(hi, x))

def win_prob(home_row, away_row) -> float:
    base = 50 + 0.5 * (home_row["overall"] - away_row["overall"])
    base += HOME_FIELD_BONUS
    base += 0.5 * (home_row["momentum"] - away_row["momentum"])
    base -= 0.5 * (home_row["injury"] - away_row["injury"])
    base += 0.1 * (home_row["qb"] - away_row["qb"])
    return clamp(base)

def simulate_score(home_row, away_row, rng: random.Random) -> Tuple[int, int]:
    """Very simple score model using offense/defense + noise."""
    base_home = 21 + (home_row["offense"] - away_row["defense"]) / 6.0
    base_away = 21 + (away_row["offense"] - home_row["defense"]) / 6.0
    # add some randomness
    home = int(max(0, rng.normalvariate(base_home, 7)))
    away = int(max(0, rng.normalvariate(base_away, 7)))
    return home, away

def simulate_season(teams_profiles: pd.DataFrame, schedule: pd.DataFrame, seed: int):
    rng = random.Random(seed)
    games_out = []
    for _, g in schedule.sort_values(["week"]).iterrows():
        home = teams_profiles.loc[teams_profiles.team_id == g.home_team_id].iloc[0]
        away = teams_profiles.loc[teams_profiles.team_id == g.away_team_id].iloc[0]
        p_home = win_prob(home, away)
        roll = rng.randint(1, 100)
        # scores
        home_pts, away_pts = simulate_score(home, away, rng)
        # ensure winner matches probability roll (override tie if needed)
        if roll <= p_home:
            if home_pts <= away_pts:
                home_pts = away_pts + 1
            winner_id = home.team_id
        else:
            if away_pts <= home_pts:
                away_pts = home_pts + 1
            winner_id = away.team_id
        games_out.append({
            "week": g.week,
            "home_team_id": home.team_id,
            "away_team_id": away.team_id,
            "home_points": home_pts,
            "away_points": away_pts,
            "win_prob_home_pct": round(p_home, 1),
            "rng_roll": roll,
            "winner_team_id": winner_id
        })
    games_df = pd.DataFrame(games_out)

    # Build long-form for standings & tiebreakers
    tmap_conf = teams_profiles.set_index("team_id")["conference"].to_dict()

    home_long = games_df[[
        "week","home_team_id","away_team_id","home_points","away_points","winner_team_id"
    ]].rename(columns={
        "home_team_id":"team_id",
        "away_team_id":"opponent_id",
        "home_points":"points_for",
        "away_points":"points_against"
    })
    home_long["is_win"] = home_long["winner_team_id"] == home_long["team_id"]

    away_long = games_df[[
        "week","home_team_id","away_points","home_points","winner_team_id","away_team_id"
    ]].rename(columns={
        "away_team_id":"team_id",
        "home_team_id":"opponent_id",
        "away_points":"points_for",
        "home_points":"points_against"
    })
    away_long["is_win"] = away_long["winner_team_id"] == away_long["team_id"]

    long = pd.concat([home_long, away_long], ignore_index=True)
    long["team_conf"] = long["team_id"].map(tmap_conf)
    long["opp_conf"] = long["opponent_id"].map(tmap_conf)
    long["is_conf_game"] = long["team_conf"] == long["opp_conf"]

    # Aggregate standings
    agg = long.groupby("team_id").agg(
        wins=("is_win","sum"),
        games=("is_win","count"),
        points_for=("points_for","sum"),
        points_against=("points_against","sum"),
        conf_wins=("is_win", lambda s: s[long.loc[s.index,"is_conf_game"]].sum()),
        conf_games=("is_conf_game","sum")
    ).reset_index()

    agg["losses"] = agg["games"] - agg["wins"]
    agg["point_diff"] = agg["points_for"] - agg["points_against"]
    agg["conf_losses"] = agg["conf_games"] - agg["conf_wins"]
    agg["conf_win_pct"] = agg["conf_wins"] / agg["conf_games"].replace(0, np.nan)

    standings = agg.merge(
        teams_profiles[["team_id","team_name","emoji","conference","division",
                        "profile_name","overall","offense","defense","qb","momentum","injury"]],
        on="team_id", how="left"
    )

    return games_df, long, standings

# -------------------------------
# Tiebreakers & Playoffs
# -------------------------------

def head_to_head_record(long: pd.DataFrame, team_a: int, team_b: int) -> Tuple[int,int]:
    mask = ((long["team_id"] == team_a) & (long["opponent_id"] == team_b)) | \
           ((long["team_id"] == team_b) & (long["opponent_id"] == team_a))
    subset = long[mask]
    a_wins = subset[(subset["team_id"] == team_a) & (subset["is_win"])].shape[0]
    b_wins = subset[(subset["team_id"] == team_b) & (subset["is_win"])].shape[0]
    return a_wins, b_wins

def apply_tiebreakers(group: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Apply tiebreakers within a wins/losses group."""
    if len(group) <= 1:
        return group

    df = group.copy()

    # 1) For 2-team ties, try head-to-head
    if len(df) == 2:
        a, b = df["team_id"].tolist()
        a_w, b_w = head_to_head_record(long, a, b)
        if a_w != b_w:
            # a gets higher rank if more H2H wins
            df = df.set_index("team_id")
            if a_w > b_w:
                return df.loc[[a, b]].reset_index()
            else:
                return df.loc[[b, a]].reset_index()

    # 2) Conference record, then point differential
    df = df.sort_values(
        ["conf_win_pct","point_diff","overall"],
        ascending=[False, False, False]
    )
    return df

def rank_conference(standings: pd.DataFrame, long: pd.DataFrame, conf: str) -> pd.DataFrame:
    conf_df = standings[standings["conference"] == conf].copy()

    ranked_segments = []
    for (wins, losses), grp in conf_df.groupby(["wins","losses"], sort=False):
        ranked_segments.append(apply_tiebreakers(grp, long))
    ranked = pd.concat(ranked_segments, ignore_index=True)

    # Now order by wins, then tiebreakers (already applied inside groups)
    ranked = ranked.sort_values(
        ["wins","conf_win_pct","point_diff","overall"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return ranked

def build_playoff_seeds(standings: pd.DataFrame, long: pd.DataFrame):
    """Return DataFrame of playoff seeds for both conferences."""
    seeds = []
    for conf in standings["conference"].unique():
        conf_ranked = rank_conference(standings, long, conf)

        # division winners
        div_winners = []
        for div in conf_ranked["division"].unique():
            div_block = conf_ranked[conf_ranked["division"] == div]
            if not div_block.empty:
                div_winners.append(div_block.iloc[0])
        div_winners_df = pd.DataFrame(div_winners)
        # mark by their index in conf_ranked
        div_winners_df["rank_index"] = div_winners_df["team_id"].map(
            {tid: i for i, tid in enumerate(conf_ranked["team_id"].tolist())}
        )
        div_winners_df = div_winners_df.sort_values("rank_index").reset_index(drop=True)

        # wildcards = next-best teams not division winners
        div_winner_ids = set(div_winners_df["team_id"].tolist())
        others = conf_ranked[~conf_ranked["team_id"].isin(div_winner_ids)].reset_index(drop=True)
        wildcards_df = others.head(3)

        # assign seeds: 1–4 division winners, 5–7 wildcards (in conf_ranked order)
        seed_no = 1
        for _, row in div_winners_df.iterrows():
            row_dict = row.to_dict()
            row_dict["seed"] = seed_no
            seeds.append(row_dict)
            seed_no += 1
        for _, row in wildcards_df.iterrows():
            row_dict = row.to_dict()
            row_dict["seed"] = seed_no
            seeds.append(row_dict)
            seed_no += 1

    seeds_df = pd.DataFrame(seeds)
    seeds_df = seeds_df.sort_values(
        ["conference","seed"], ascending=[True, True]
    ).reset_index(drop=True)
    return seeds_df

def simulate_playoffs(seeds_df: pd.DataFrame, teams_profiles: pd.DataFrame, base_seed: int = 10000):
    """Simulate full playoffs (both conferences + Super Bowl)."""
    rng = random.Random(base_seed)
    tp = teams_profiles.set_index("team_id")

    playoff_games = []

    def play_series(conf: str, round_name: str, matchups: List[Tuple[int,int]]) -> List[int]:
        winners = []
        for high_seed, low_seed in matchups:
            row_hi = seeds_df[(seeds_df["conference"] == conf) & (seeds_df["seed"] == high_seed)].iloc[0]
            row_lo = seeds_df[(seeds_df["conference"] == conf) & (seeds_df["seed"] == low_seed)].iloc[0]
            team_hi = tp.loc[row_hi["team_id"]]
            team_lo = tp.loc[row_lo["team_id"]]

            # higher seed is home
            home_row, away_row = team_hi, team_lo
            home_seed, away_seed = high_seed, low_seed
            p_home = win_prob(home_row, away_row)
            roll = rng.randint(1, 100)
            home_pts, away_pts = simulate_score(home_row, away_row, rng)
            if roll <= p_home:
                if home_pts <= away_pts:
                    home_pts = away_pts + 1
                winner_id = home_row["team_id"]
                winner_seed = home_seed
            else:
                if away_pts <= home_pts:
                    away_pts = home_pts + 1
                winner_id = away_row["team_id"]
                winner_seed = away_seed

            playoff_games.append({
                "round": round_name,
                "conference": conf,
                "home_team_id": home_row["team_id"],
                "away_team_id": away_row["team_id"],
                "home_points": home_pts,
                "away_points": away_pts,
                "win_prob_home_pct": round(p_home, 1),
                "rng_roll": roll,
                "winner_team_id": winner_id
            })
            winners.append(winner_seed)
        return winners

    conf_champs = {}

    for conf in seeds_df["conference"].unique():
        conf_seeds = seeds_df[seeds_df["conference"] == conf]

        # Wild Card: 2 vs 7, 3 vs 6, 4 vs 5
        wc_matchups = [(2,7),(3,6),(4,5)]
        wc_winners = play_series(conf, "WC", wc_matchups)

        # Divisional: 1 vs lowest remaining, others play
        lowest = min(wc_winners)
        others = sorted([s for s in wc_winners if s != lowest])
        div_matchups = [(1, lowest), (others[0], others[1])]
        div_winners = play_series(conf, "DIV", div_matchups)

        # Conference Championship
        cc_matchup = [(min(div_winners), max(div_winners))]
        cc_winners = play_series(conf, "CONF", cc_matchup)
        conf_champs[conf] = cc_winners[0]

    # Super Bowl: champs from each conference
    confs = sorted(conf_champs.keys())
    conf_a, conf_b = confs[0], confs[1]
    seed_a = conf_champs[conf_a]
    seed_b = conf_champs[conf_b]

    row_a = seeds_df[(seeds_df["conference"] == conf_a) & (seeds_df["seed"] == seed_a)].iloc[0]
    row_b = seeds_df[(seeds_df["conference"] == conf_b) & (seeds_df["seed"] == seed_b)].iloc[0]
    team_a = tp.loc[row_a["team_id"]]
    team_b = tp.loc[row_b["team_id"]]

    # For SB, arbitrary: conf with lower name is "home"
    if conf_a < conf_b:
        home_row, away_row = team_a, team_b
        home_conf, away_conf = conf_a, conf_b
    else:
        home_row, away_row = team_b, team_a
        home_conf, away_conf = conf_b, conf_a

    p_home = win_prob(home_row, away_row)
    roll = rng.randint(1, 100)
    home_pts, away_pts = simulate_score(home_row, away_row, rng)
    if roll <= p_home:
        if home_pts <= away_pts:
            home_pts = away_pts + 1
        winner_id = home_row["team_id"]
        sb_winner_conf = home_conf
    else:
        if away_pts <= home_pts:
            away_pts = home_pts + 1
        winner_id = away_row["team_id"]
        sb_winner_conf = away_conf

    playoff_games.append({
        "round": "SB",
        "conference": "NFL",
        "home_team_id": home_row["team_id"],
        "away_team_id": away_row["team_id"],
        "home_points": home_pts,
        "away_points": away_pts,
        "win_prob_home_pct": round(p_home, 1),
        "rng_roll": roll,
        "winner_team_id": winner_id
    })

    playoff_df = pd.DataFrame(playoff_games)
    return playoff_df

# -------------------------------
# UI – Main layout
# -------------------------------

st.title("🏈 HELMETS – League Manager")

with st.sidebar:
    st.header("Setup")

    source = st.radio(
        "Teams / Profiles Source",
        ["Google Sheet", "Upload CSVs", "Demo"],
        index=0
    )

    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Initialize league"):
        teams_df = None
        profiles_df = None
        try:
            if source == "Google Sheet":
                teams_df = read_sheet("Teams")
                profiles_df = read_sheet("Profiles")
            elif source == "Upload CSVs":
                st.warning("Use the upload widgets on the main page, then click this again.")
                st.stop()
            elif source == "Demo":
                teams_df = make_demo_teams()
                profiles_df = make_demo_profiles(50)

            if teams_df is not None and profiles_df is not None:
                prof_pool = profiles_df.sample(
                    n=len(teams_df), replace=True, random_state=int(seed)
                ).reset_index(drop=True)
                teams_prof = teams_df.copy()
                teams_prof["profile_id"] = prof_pool["profile_id"].values
                teams_prof = teams_prof.merge(profiles_df, on="profile_id", how="left")
                sched_df = generate_schedule(teams_df, seed=int(seed))

                st.session_state["source"] = source
                st.session_state["seed"] = int(seed)
                st.session_state["teams"] = teams_df
                st.session_state["profiles"] = profiles_df
                st.session_state["teams_profiles"] = teams_prof
                st.session_state["schedule"] = sched_df
                st.session_state["games"] = None
                st.session_state["standings"] = None
                st.session_state["playoff_seeds"] = None
                st.session_state["playoffs"] = None
                st.success("League initialized.")
        except Exception as e:
            st.error(f"Error initializing league: {e}")

# Main state
teams = st.session_state.get("teams")
profiles = st.session_state.get("profiles")
teams_profiles = st.session_state.get("teams_profiles")
schedule = st.session_state.get("schedule")

# -------------------------------
# CSV upload (for Upload mode)
# -------------------------------
st.markdown("### Upload CSVs (optional)")

cols_up = st.columns(2)
with cols_up[0]:
    teams_up = st.file_uploader("Upload teams.csv", type=["csv"], key="teams_upload")
with cols_up[1]:
    profiles_up = st.file_uploader("Upload profiles.csv", type=["csv"], key="profiles_upload")

if teams_up is not None:
    teams_from_csv = pd.read_csv(teams_up, encoding="utf-8", engine="python")
    st.session_state["teams"] = teams_from_csv
    teams = teams_from_csv
    st.success(f"Loaded {len(teams_from_csv)} teams from CSV.")
if profiles_up is not None:
    profiles_from_csv = pd.read_csv(profiles_up, encoding="utf-8", engine="python")
    st.session_state["profiles"] = profiles_from_csv
    profiles = profiles_from_csv
    st.success(f"Loaded {len(profiles_from_csv)} profiles from CSV.")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "League & Profiles",
    "Teams Editor",
    "Profiles Editor",
    "Schedule",
    "Simulate Season & Playoffs"
])

with tab1:
    st.subheader("Teams + Assigned Profiles")
    if teams_profiles is None:
        st.info("Initialize the league from the sidebar.")
    else:
        st.dataframe(
            teams_profiles.sort_values(["conference","division","team_name"])
            [["team_id","team_name","emoji","conference","division",
              "profile_name","overall","offense","defense","qb","momentum","injury"]],
            use_container_width=True, height=500
        )

with tab2:
    st.subheader("Edit Teams")
    if teams is None:
        st.info("No teams loaded yet. Initialize from sidebar or upload teams.csv.")
    else:
        st.caption("Edit names, emojis, conferences and divisions. "
                   "Use 'Save Teams' to update state/Sheets.")
        edited_teams = st.data_editor(
            teams,
            num_rows="dynamic",
            use_container_width=True,
            key="teams_editor"
        )

        cols_btn = st.columns(3)
        with cols_btn[0]:
            if st.button("Save Teams (App Only)"):
                st.session_state["teams"] = edited_teams
                st.success("Teams updated in app state. Rebuild league to reassign profiles & schedule.")

        with cols_btn[1]:
            if st.button("Save Teams to Google Sheet"):
                try:
                    upsert_sheet("Teams", edited_teams)
                    st.session_state["teams"] = edited_teams
                    st.success("Teams saved to Google Sheet 'Teams' and app state updated.")
                except Exception as e:
                    st.error(f"Failed to save to Google Sheet: {e}")

        with cols_btn[2]:
            if st.button("Rebuild League From Current Teams & Profiles"):
                if profiles is None:
                    st.error("No profiles loaded yet.")
                else:
                    prof_pool = profiles.sample(
                        n=len(edited_teams), replace=True,
                        random_state=int(st.session_state.get("seed", 42))
                    ).reset_index(drop=True)
                    teams_prof = edited_teams.copy()
                    teams_prof["profile_id"] = prof_pool["profile_id"].values
                    teams_prof = teams_prof.merge(profiles, on="profile_id", how="left")
                    sched_df = generate_schedule(edited_teams, seed=int(st.session_state.get("seed", 42)))

                    st.session_state["teams"] = edited_teams
                    st.session_state["teams_profiles"] = teams_prof
                    st.session_state["schedule"] = sched_df
                    st.session_state["games"] = None
                    st.session_state["standings"] = None
                    st.session_state["playoff_seeds"] = None
                    st.session_state["playoffs"] = None
                    st.success("League rebuilt with updated teams.")

with tab3:
    st.subheader("Edit Profiles")
    if profiles is None:
        st.info("No profiles loaded yet. Initialize from sidebar, use demo, or upload profiles.csv.")
    else:
        st.caption("Edit profile ratings and descriptions. Then save to app and/or Google Sheets.")
        edited_profiles = st.data_editor(
            profiles,
            num_rows="dynamic",
            use_container_width=True,
            key="profiles_editor"
        )

        cols_p = st.columns(3)
        with cols_p[0]:
            if st.button("Save Profiles (App Only)"):
                st.session_state["profiles"] = edited_profiles
                st.success("Profiles updated in app state. Rebuild league to reassign profiles.")
        with cols_p[1]:
            if st.button("Save Profiles to Google Sheet"):
                try:
                    upsert_sheet("Profiles", edited_profiles)
                    st.session_state["profiles"] = edited_profiles
                    st.success("Profiles saved to Google Sheet 'Profiles' and app state updated.")
                except Exception as e:
                    st.error(f"Failed to save Profiles to Google Sheet: {e}")
        with cols_p[2]:
            if st.button("Rebuild League From Current Profiles & Teams"):
                if teams is None:
                    st.error("No teams loaded yet.")
                else:
                    prof_pool = edited_profiles.sample(
                        n=len(teams), replace=True,
                        random_state=int(st.session_state.get("seed", 42))
                    ).reset_index(drop=True)
                    teams_prof = teams.copy()
                    teams_prof["profile_id"] = prof_pool["profile_id"].values
                    teams_prof = teams_prof.merge(edited_profiles, on="profile_id", how="left")
                    sched_df = generate_schedule(teams, seed=int(st.session_state.get("seed", 42)))

                    st.session_state["profiles"] = edited_profiles
                    st.session_state["teams_profiles"] = teams_prof
                    st.session_state["schedule"] = sched_df
                    st.session_state["games"] = None
                    st.session_state["standings"] = None
                    st.session_state["playoff_seeds"] = None
                    st.session_state["playoffs"] = None
                    st.success("League rebuilt with updated profiles.")

with tab4:
    st.subheader("Schedule")
    if schedule is None:
        st.info("Initialize the league to generate a schedule.")
    else:
        st.dataframe(
            schedule.sort_values(["week","home_team_id"]),
            use_container_width=True,
            height=500
        )

with tab5:
    st.subheader("Simulate Season & Playoffs")

    if teams_profiles is None or schedule is None:
        st.info("Initialize the league first.")
    else:
        sim_seed = st.number_input(
            "Simulation seed",
            value=int(st.session_state.get("seed", 42)),
            step=1
        )
        if st.button("Run Regular Season + Playoffs"):
            games_df, long_df, standings_df = simulate_season(
                teams_profiles, schedule, seed=int(sim_seed)
            )
            playoff_seeds = build_playoff_seeds(standings_df, long_df)
            playoffs_df = simulate_playoffs(playoff_seeds, teams_profiles, base_seed=int(sim_seed) + 1000)

            st.session_state["games"] = games_df
            st.session_state["long"] = long_df
            st.session_state["standings"] = standings_df
            st.session_state["playoff_seeds"] = playoff_seeds
            st.session_state["playoffs"] = playoffs_df

        standings = st.session_state.get("standings")
        playoff_seeds = st.session_state.get("playoff_seeds")
        playoffs = st.session_state.get("playoffs")
        games = st.session_state.get("games")

        if standings is not None:
            st.markdown("### Regular Season Standings (Top 10)")
            st.dataframe(
                standings.sort_values(["wins","point_diff"], ascending=[False, False])
                .head(10)[
                    ["team_name","emoji","conference","division","wins","losses",
                     "points_for","points_against","point_diff",
                     "conf_wins","conf_losses","profile_name"]
                ],
                use_container_width=True,
                height=350
            )

        if playoff_seeds is not None:
            st.markdown("### Playoff Seeding")
            st.dataframe(
                playoff_seeds[[
                    "conference","seed","team_name","emoji","division",
                    "wins","losses","conf_wins","conf_losses","point_diff"
                ]].sort_values(["conference","seed"]),
                use_container_width=True,
                height=350
            )

        if playoffs is not None:
            st.markdown("### Playoff Results")
            tmap = teams_profiles.set_index("team_id")["team_name"].to_dict()
            pshow = playoffs.copy()
            pshow["home_team"] = pshow["home_team_id"].map(tmap)
            pshow["away_team"] = pshow["away_team_id"].map(tmap)
            pshow["winner_team"] = pshow["winner_team_id"].map(tmap)
            pshow = pshow[
                ["round","conference","home_team","away_team",
                 "home_points","away_points","winner_team","win_prob_home_pct","rng_roll"]
            ]
            st.dataframe(
                pshow.sort_values(["round","conference"]),
                use_container_width=True,
                height=400
            )

        if standings is not None:
            buf_s = io.StringIO()
            standings.to_csv(buf_s, index=False)
            st.download_button("Download Standings CSV", buf_s.getvalue(),
                               file_name="standings.csv", mime="text/csv")

        if games is not None:
            st.markdown("### All Regular Season Games")
            tmap = teams_profiles.set_index("team_id")["team_name"].to_dict()
            gshow = games.copy()
            gshow["home_team"] = gshow["home_team_id"].map(tmap)
            gshow["away_team"] = gshow["away_team_id"].map(tmap)
            gshow["winner_team"] = gshow["winner_team_id"].map(tmap)
            gshow = gshow[
                ["week","home_team","away_team",
                 "home_points","away_points",
                 "win_prob_home_pct","rng_roll","winner_team"]
            ].sort_values(["week"])
            st.dataframe(gshow, use_container_width=True, height=350)

            buf_g = io.StringIO()
            gshow.to_csv(buf_g, index=False)
            st.download_button("Download Games CSV", buf_g.getvalue(),
                               file_name="games.csv", mime="text/csv")

st.markdown("---")
st.caption("HELMETS – Teams & Profiles editable via app or Google Sheets. Includes regular season, tiebreakers, and full playoff bracket.")