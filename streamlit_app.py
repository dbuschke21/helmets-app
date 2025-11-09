import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import io
import random

st.set_page_config(page_title="HELMETS – MVP", layout="wide")

# -------------------------------
# Utilities / Data structures
# -------------------------------

@dataclass
class Profile:
    profile_id: int
    profile_name: str
    overall: int
    offense: int
    defense: int
    qb: int
    momentum: int          # simple scalar for MVP; later: list[17]
    injury: int            # simple scalar for MVP; later: list[17]
    description: str

HOME_FIELD_BONUS = 3.0

# -------------------------------
# Session helpers
# -------------------------------

def get_ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# -------------------------------
# Demo league + profile factories
# -------------------------------

DIVS = ["East", "North", "South", "West"]
EMOJIS = ["🏈","🦅","🐻","🐯","🦬","🦁","🐺","🐻‍❄️","🐬","🐻‍🚒","🦈","🔥","⚡","🌪️","🌊","❄️"]

def make_demo_teams() -> pd.DataFrame:
    teams = []
    # 2 conferences x 4 divisions x 4 teams = 32
    confs = ["AFC", "NFC"]
    tid = 1
    for conf in confs:
        for d in DIVS:
            for i in range(4):
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
        qb      = int(np.clip(rng.normal((overall-60)/ (99-60) * 28 + 4, 6), 0, 32))
        momentum= int(np.clip(rng.integers(0, 9), 0, 9))
        injury  = int(np.clip(rng.integers(0, 6), 0, 9))
        desc = "Auto-generated profile"
        rows.append([pid, f"Profile {pid}", overall, offense, defense, qb, momentum, injury, desc])
    return pd.DataFrame(rows, columns=[
        "profile_id","profile_name","overall","offense","defense","qb","momentum","injury","description"
    ])

# -------------------------------
# Schedule generator – follows your rules
# 1) No divisional games
# 2) Play all other conference opponents once (12 games)
# 3) Each division plays all 4 teams from one cross-conf division (4)
# 4) +1 random cross-conf opponent (1)
# Total = 17
# -------------------------------

def schedule_for_team(team_id: int, teams: pd.DataFrame, cross_conf_map: Dict[str,str]) -> List[Tuple[int,int]]:
    """Return list of (home_id, away_id) opponents for a single team, length 17.
       Later we will resolve home/away balance; for MVP we’ll balance roughly.
    """
    row = teams.loc[teams.team_id == team_id].iloc[0]
    conf, div = row.conference, row.division

    # Intra-conference (12): all non-divisional opponents
    intra_conf = teams[(teams.conference == conf) & (teams.division != div)]
    intra_ids = intra_conf.team_id.tolist()  # 3 divisions x 4 teams = 12

    # Cross-conference division (4):
    cross_div = cross_conf_map[div]
    cross_conf = "NFC" if conf == "AFC" else "AFC"
    cc_div_teams = teams[(teams.conference == cross_conf) & (teams.division == cross_div)]
    cc_div_ids = cc_div_teams.team_id.tolist()  # 4

    # One random cross-conf opponent (1) not already included
    others = teams[(teams.conference == cross_conf) & (teams.division != cross_div)]
    others_ids = others.team_id.tolist()
    rnd = random.choice(others_ids)

    opponents = intra_ids + cc_div_ids + [rnd]  # 12 + 4 + 1 = 17
    assert len(opponents) == 17

    # Create home/away roughly 50/50; alternate
    pairs = []
    flip = True
    for opp in opponents:
        if flip:
            pairs.append((team_id, opp))   # home
        else:
            pairs.append((opp, team_id))   # away
        flip = not flip
    return pairs

def generate_schedule(teams: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    # Map divisions across conferences: East<->East, etc.
    cross_conf_map = {d:d for d in DIVS}

    # Build team-specific desired opponents
    desired = {tid: schedule_for_team(tid, teams, cross_conf_map) for tid in teams.team_id}

    # To avoid duplicates and clashes, we’ll assemble week-by-week with a greedy matcher.
    # For MVP: simple pairing across all desired games; if duplicates appear, we only keep once.
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

    # We should end up near (32 teams * 17) / 2 = 272 games
    # If a few are missing due to collisions, we won't stress in MVP.
    # Assign weeks by shuffling and slicing into 17 rounds.
    random.shuffle(all_games)
    weeks = []
    n_weeks = 17
    # naive packer: distribute games evenly across weeks
    per_week = len(all_games) // n_weeks
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

# -------------------------------
# Win probability & simulation
# -------------------------------

def clamp(x, lo=10, hi=90):
    return max(lo, min(hi, x))

def win_prob(home_row, away_row) -> float:
    # Base from overall gap
    base = 50 + 0.5 * (home_row["overall"] - away_row["overall"])
    # Home field
    base += HOME_FIELD_BONUS
    # Momentum and injury (simple scalar MVP)
    base += 0.5 * (home_row["momentum"] - away_row["momentum"])
    base -= 0.5 * (home_row["injury"] - away_row["injury"])
    # QB adds volatility: push slightly toward the higher QB
    base += 0.1 * (home_row["qb"] - away_row["qb"])
    return clamp(base)

def simulate_season(teams_profiles: pd.DataFrame, schedule: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    games_out = []
    for _, g in schedule.sort_values(["week"]).iterrows():
        home = teams_profiles.loc[teams_profiles.team_id == g.home_team_id].iloc[0]
        away = teams_profiles.loc[teams_profiles.team_id == g.away_team_id].iloc[0]
        p_home = win_prob(home, away)
        roll = rng.randint(1, 100)
        winner_id = home.team_id if roll <= p_home else away.team_id
        games_out.append({
            "week": g.week,
            "home_team_id": home.team_id,
            "away_team_id": away.team_id,
            "win_prob_home_pct": round(p_home, 1),
            "rng_roll": roll,
            "winner_team_id": winner_id
        })
    games_df = pd.DataFrame(games_out)

    # Standings
    home_side = games_df.rename(columns={"home_team_id":"team_id"})
    home_side["is_win"] = home_side["winner_team_id"] == home_side["team_id"]
    away_side = games_df.rename(columns={"away_team_id":"team_id"})
    away_side["is_win"] = away_side["winner_team_id"] == away_side["team_id"]

    tall = pd.concat([home_side[["team_id","is_win"]], away_side[["team_id","is_win"]]])
    standings = tall.groupby("team_id")["is_win"].agg(["sum","count"]).reset_index()
    standings.columns = ["team_id","wins","games"]
    standings["losses"] = standings["games"] - standings["wins"]
    standings = standings.drop(columns=["games"])

    # Join names and metadata
    standings = standings.merge(
        teams_profiles[["team_id","team_name","emoji","conference","division",
                        "profile_name","overall","offense","defense","qb","momentum","injury"]],
        on="team_id", how="left"
    ).sort_values(["wins","overall"], ascending=[False, False])

    return games_df, standings

# -------------------------------
# UI
# -------------------------------

st.title("🏈 HELMETS — Streamlit MVP")
st.caption("Draft-era NFL sim with probabilistic outcomes. This MVP: load teams, auto-generate schedule, assign profiles, simulate season, show standings.")

with st.sidebar:
    st.header("Setup")
    seed = st.number_input("Random seed", value=get_ss("seed", 42), step=1)
    if st.button("Reset session state"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.subheader("Load data")
    teams_file = st.file_uploader("teams.csv (team_id,team_name,emoji,conference,division)", type=["csv"])
    profiles_file = st.file_uploader("profiles.csv (optional)", type=["csv"])

    st.markdown("---")
    use_demo = st.checkbox("Use demo 32-team league", value=(teams_file is None))
    if st.button("Initialize league"):
        random.seed(seed)
        # Load or demo teams
        if use_demo:
            teams_df = make_demo_teams()
        else:
            if teams_file is None:
                st.error("Please upload teams.csv or toggle demo league.")
                st.stop()
            teams_df = pd.read_csv(teams_file)
            required_cols = {"team_id","team_name","emoji","conference","division"}
            if not required_cols.issubset(set(teams_df.columns)):
                st.error(f"teams.csv missing columns: {required_cols - set(teams_df.columns)}")
                st.stop()

        # Load or demo profiles
        if profiles_file is not None:
            profiles_df = pd.read_csv(profiles_file)
        else:
            profiles_df = make_demo_profiles(50)

        # Assign profiles randomly (with replacement; many more profiles is fine)
        prof_pool = profiles_df.sample(n=len(teams_df), replace=True, random_state=seed).reset_index(drop=True)
        teams_prof = teams_df.copy()
        teams_prof["profile_id"] = prof_pool["profile_id"].values
        teams_prof = teams_prof.merge(profiles_df, on="profile_id", how="left")

        # Schedule
        sched_df = generate_schedule(teams_df, seed=seed)

        st.session_state["teams"] = teams_df
        st.session_state["profiles"] = profiles_df
        st.session_state["teams_profiles"] = teams_prof
        st.session_state["schedule"] = sched_df
        st.session_state["seed"] = int(seed)
        st.success("League initialized!")

# Main panels
teams_profiles = st.session_state.get("teams_profiles")
schedule = st.session_state.get("schedule")

tab1, tab2, tab3 = st.tabs(["League & Profiles", "Schedule", "Simulate Season"])

with tab1:
    st.subheader("Teams + Assigned Profiles")
    if teams_profiles is None:
        st.info("Initialize the league in the sidebar.")
    else:
        st.dataframe(
            teams_profiles.sort_values(["conference","division","team_name"])
            [["team_id","team_name","emoji","conference","division",
              "profile_name","overall","offense","defense","qb","momentum","injury"]],
            use_container_width=True, height=500
        )

with tab2:
    st.subheader("Generated Schedule (MVP)")
    if schedule is None:
        st.info("Initialize the league in the sidebar.")
    else:
        st.caption("Rules: 12 intra-conference non-divisional, 4 vs one cross-conf division, 1 random cross-conf. Home/away alternates per team (approx).")
        st.dataframe(schedule.sort_values(["week","home_team_id"]), use_container_width=True, height=500)

with tab3:
    st.subheader("Run Simulation")
    if teams_profiles is None or schedule is None:
        st.info("Initialize the league in the sidebar.")
    else:
        sim_seed = st.number_input("Simulation seed (deterministic)", value=get_ss("seed", 42), step=1)
        if st.button("Sim Full Season"):
            games_df, standings_df = simulate_season(teams_profiles, schedule, seed=int(sim_seed))
            st.session_state["games"] = games_df
            st.session_state["standings"] = standings_df

        games = st.session_state.get("games")
        st.markdown("### Standings")
        standings = st.session_state.get("standings")
        if standings is not None:
            # Conference/Division group view
            with st.expander("By Conference / Division"):
                grouped = standings.sort_values(["conference","division","wins","overall"], ascending=[True, True, False, False])
                st.dataframe(grouped[["team_name","emoji","conference","division","wins","losses","overall","qb","profile_name"]],
                             use_container_width=True, height=420)

            st.markdown("**Top 10 Overall**")
            st.dataframe(standings.head(10)[["team_name","emoji","wins","losses","overall","qb","profile_name"]],
                         use_container_width=True, height=320)

            # Download buttons
            buf_s = io.StringIO()
            standings.to_csv(buf_s, index=False)
            st.download_button("Download Standings CSV", buf_s.getvalue(), file_name="standings.csv", mime="text/csv")

        st.markdown("### Games (All)")
        if games is not None:
            # Join team names for readability
            tmap = teams_profiles.set_index("team_id")["team_name"].to_dict()
            gshow = games.copy()
            gshow["home_team"] = gshow["home_team_id"].map(tmap)
            gshow["away_team"] = gshow["away_team_id"].map(tmap)
            gshow["winner_team"] = gshow["winner_team_id"].map(tmap)
            gshow = gshow[["week","home_team","away_team","win_prob_home_pct","rng_roll","winner_team"]].sort_values(["week"])
            st.dataframe(gshow, use_container_width=True, height=400)

            buf_g = io.StringIO()
            gshow.to_csv(buf_g, index=False)
            st.download_button("Download Games CSV", buf_g.getvalue(), file_name="games.csv", mime="text/csv")

st.markdown("---")
st.caption("MVP notes: schedule packing is greedy; home/away is approximate; momentum/injury are scalar. Next steps: weekly curves, tie-breakers, playoffs, GOW, and Realtime multiplayer.")