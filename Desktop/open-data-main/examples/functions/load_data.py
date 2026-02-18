import json
import re
import pandas as pd

# define function to load matches
def load_matches(iteration_id: int = 743) -> pd.DataFrame:

    # load matches
    with open(f"../data/matches/matches_{iteration_id}.json", "r") as f:
        matches_data = json.load(f)
    matches = pd.json_normalize(matches_data)

    # fix column names using regex
    matches = matches.rename(columns=lambda x: re.sub(r"\.(.)", lambda y: y.group(1).upper(), x))

    # load squads
    squads = pd.read_json(f"../data/squads/squads_{iteration_id}.json")

    # join matches and squads
    matches = matches.merge(squads[["id", "name"]].rename(
        columns={"id": "homeSquadId", "name": "homeSquadName"}),
        left_on="homeSquadId",
        right_on="homeSquadId"
    )
    matches = matches.merge(squads[["id", "name"]].rename(
        columns={"id": "awaySquadId", "name": "awaySquadName"}),
        left_on="awaySquadId",
        right_on="awaySquadId"
    )

    # set order
    matches = matches[[
        "iterationId",
        "id",
        "matchDayIndex",
        "matchDayName",
        "homeSquadId",
        "homeSquadName",
        "awaySquadId",
        "awaySquadName",
        "scheduledDate",
        "lastCalculationDate",
        "available"
    ]]

    return matches


# define function to load event data for a list of matches
def load_match_events(iteration_id: int = 743, match_ids = None) -> pd.DataFrame:

    # handle default case
    if match_ids is None:
        match_ids = [122839, 122840, 122841]

    # load iteration master data
    players = pd.read_json(f"../data/players/players_{iteration_id}.json")
    squads = pd.read_json(f"../data/squads/squads_{iteration_id}.json")
    matches = load_matches(iteration_id=iteration_id)

    # create empty list
    events = []

    # iterate over matches
    for match_id in match_ids:

        # load match events
        with open(f"../data/events/events_{match_id}.json", "r") as f:
            match_events_data = json.load(f)
        match_events = pd.json_normalize(match_events_data).assign(matchId=match_id)

        # append to events df
        events.append(match_events)

    # convert events to df
    events = pd.concat(events)

    # fix column names using regex
    events = events.rename(columns=lambda x: re.sub(r"\.(.)", lambda y: y.group(1).upper(), x))

    # join data
    events = events.merge(
        squads[["id", "name"]].rename(columns={"id": "squadId", "name": "squadName"}),
        left_on="squadId",
        right_on="squadId",
        how="left",
        suffixes=("", "_home")
    ).merge(
        squads[["id", "name"]].rename(columns={"id": "squadId", "name": "currentAttackingSquadName"}),
        left_on="currentAttackingSquadId",
        right_on="squadId",
        how="left",
        suffixes=("", "_away")
    ).merge(
        players[["id", "commonname"]].rename(columns={"id": "playerId", "commonname": "playerName"}),
        left_on="playerId",
        right_on="playerId",
        how="left",
        suffixes=("", "_right")
    ).merge(
        players[["id", "commonname"]].rename(
            columns={"id": "pressingPlayerId", "commonname": "pressingPlayerName"}),
        left_on="pressingPlayerId",
        right_on="pressingPlayerId",
        how="left",
        suffixes=("", "_right")
    ).merge(
        players[["id", "commonname"]].rename(columns={"id": "fouledPlayerId", "commonname": "fouledPlayerName"}),
        left_on="fouledPlayerId",
        right_on="fouledPlayerId",
        how="left",
        suffixes=("", "_right")
    ).merge(
        players[["id", "commonname"]].rename(columns={"id": "duelPlayerId", "commonname": "duelPlayerName"}),
        left_on="duelPlayerId",
        right_on="duelPlayerId",
        how="left",
        suffixes=("", "_right")
    ).merge(
        players[["id", "commonname"]].rename(
            columns={"id": "passReceiverPlayerId", "commonname": "passReceiverPlayerName"}),
        left_on="passReceiverPlayerId",
        right_on="passReceiverPlayerId",
        how="left",
        suffixes=("", "_right")
    ).merge(
        matches,
        left_on="matchId",
        right_on="id",
        how="left",
        suffixes=("", "_right")
    )

    # rename some columns
    events = events.rename(columns={
        "currentAttackingSquadId": "attackingSquadId",
        "currentAttackingSquadName": "attackingSquadName",
        "duelDuelType": "duelType",
        "scheduledDate": "dateTime",
        "gameTimeGameTime": "gameTime",
        "gameTimeGameTimeInSec": "gameTimeInSec",
        "eventId": "eventId_scorings",
        "id": "eventId",
        "index": "eventNumber",
        "phaseIndex": "setPiecePhaseIndex",
        "setPieceMainEvent": "setPieceSubPhaseMainEvent",
    })

    # set eventId as index
    events = events.set_index("eventId", drop=True)

    # define desired column order
    event_cols = [
        "matchId",
        "dateTime",
        "iterationId",
        "matchDayIndex",
        "matchDayName",
        "homeSquadId",
        "homeSquadName",
        "awaySquadId",
        "awaySquadName",
        "eventNumber",
        "sequenceIndex",
        "periodId",
        "gameTime",
        "gameTimeInSec",
        "duration",
        "squadId",
        "squadName",
        "attackingSquadId",
        "attackingSquadName",
        "phase",
        "playerId",
        "playerName",
        "playerPosition",
        "playerPositionSide",
        "actionType",
        "action",
        "bodyPart",
        "bodyPartExtended",
        "previousPassHeight",
        "result",
        "startCoordinatesX",
        "startCoordinatesY",
        "startAdjCoordinatesX",
        "startAdjCoordinatesY",
        "startPackingZone",
        "startPitchPosition",
        "startLane",
        "endCoordinatesX",
        "endCoordinatesY",
        "endAdjCoordinatesX",
        "endAdjCoordinatesY",
        "endPackingZone",
        "endPitchPosition",
        "endLane",
        "opponents",
        "pressure",
        "distanceToGoal",
        "pxTTeam",
        "pxTOpponent",
        "pressingPlayerId",
        "pressingPlayerName",
        "distanceToOpponent",
        "passReceiverType",
        "passReceiverPlayerId",
        "passReceiverPlayerName",
        "passDistance",
        "passAngle",
        "shotDistance",
        "shotAngle",
        "shotTargetPointY",
        "shotTargetPointZ",
        "duelType",
        "duelPlayerId",
        "duelPlayerName",
        "fouledPlayerId",
        "fouledPlayerName",
        "formationTeam",
        "formationOpponent",
        "inferredSetPiece",
    ]

    # reorder data
    events = events[event_cols]

    return events


# define function to load event scorings for a list of matches
def load_match_scorings(iteration_id: int = 743, match_ids = None) -> pd.DataFrame:

    # handle default case
    if match_ids is None:
        match_ids = [122839, 122840, 122841]

    # load kpi definitions
    kpis = pd.read_json("../data/kpi_definitions.json")

    # create empty list
    scorings = []

    # iterate over matches
    for match_id in match_ids:

        # load match events
        with open(f"../data/events_kpis/events_kpis_{match_id}.json", "r") as f:
            match_scorings_data = json.load(f)
        match_scorings = pd.json_normalize(match_scorings_data).assign(matchId=match_id)

        # append to events df
        scorings.append(match_scorings)

    # convert events to df
    scorings = pd.concat(scorings)

    # join scorings with kpi definitions
    scorings = scorings.merge(
        kpis[["id", "name"]].rename(columns={"id": "kpiId", "name": "kpiName"}),
        on="kpiId",
        how="left"
    )

    # reorder data
    scorings = scorings[[
        "matchId",
        "eventId",
        "playerId",
        "position",
        "kpiId",
        "kpiName",
        "value"
    ]]

    return scorings