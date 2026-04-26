// Hardcoded action-space definitions for visualization.
//
// NOTE: This mirrors the discrete action space setup in
// `src/hermes/env/engine/dynamics_model/jerk_pnc_model.py` using constants from
// `src/hermes/env/constants.py`.
//
// For now we only support the jerk PNC action space: a cartesian product of
// (jerk_long, jerk_lat), where each axis is a linspace plus an ensured 0 entry.

export type JerkAxis = {
    values: number[]
    min: number
    max: number
    steps: number
}

export type JerkPncActionSpace = {
    jerkLong: number[]
    jerkLat: number[]
    nLong: number
    nLat: number
    size: number
}

export type JerkPncActionSpaceConfig = {
    minJerkLong: number
    maxJerkLong: number
    numJerkLongActions: number
    minJerkLat: number
    maxJerkLat: number
    numJerkLatActions: number
    lateralActionSpaceShapeGamma?: number
}

function linspace(start: number, end: number, steps: number): number[] {
    if (steps <= 0) return []
    if (steps === 1) return [start]
    const out: number[] = []
    const step = (end - start) / (steps - 1)
    for (let i = 0; i < steps; i++) out.push(start + i * step)
    return out
}

function ensureZeroInAxisSorted(axis: number[]): number[] {
    // The Python code uses `if 0 not in tensor` which is exact; in JS floats we
    // use a tiny tolerance.
    const EPS = 1e-12
    const hasZero = axis.some((v) => Math.abs(v) <= EPS)
    const out = hasZero ? [...axis] : [...axis, 0]
    out.sort((a, b) => a - b)
    return out
}

/**
 * Hardcoded jerk PNC action space.
 *
 * constants.py:
 * - MIN_JERK_LONG=-1.5, MAX_JERK_LONG=1.5, NUM_JERK_LONG_ACTIONS=12
 * - MIN_JERK_LAT=-1.0, MAX_JERK_LAT=1.0, NUM_JERK_LAT_ACTIONS=12
 */
export function getJerkPncActionSpace(): JerkPncActionSpace {
    const jerkLong = ensureZeroInAxisSorted(linspace(-1.5, 1.5, 12))
    const jerkLat = ensureZeroInAxisSorted(linspace(-1.0, 1.0, 20))

    return {
        jerkLong,
        jerkLat,
        nLong: jerkLong.length,
        nLat: jerkLat.length,
        size: jerkLong.length * jerkLat.length
    }
}

/**
 * Create jerk PNC action space from runtime config (typically from backend /env).
 * Falls back to defaults when config is missing or invalid.
 */
export function getJerkPncActionSpaceFromConfig(config?: Partial<JerkPncActionSpaceConfig> | null): JerkPncActionSpace {
    const minJerkLong = Number(config?.minJerkLong)
    const maxJerkLong = Number(config?.maxJerkLong)
    const numJerkLongActions = Math.trunc(Number(config?.numJerkLongActions))
    const minJerkLat = Number(config?.minJerkLat)
    const maxJerkLat = Number(config?.maxJerkLat)
    const numJerkLatActions = Math.trunc(Number(config?.numJerkLatActions))
    const gamma = Number(config?.lateralActionSpaceShapeGamma)

    const valid =
        Number.isFinite(minJerkLong) &&
        Number.isFinite(maxJerkLong) &&
        Number.isFinite(minJerkLat) &&
        Number.isFinite(maxJerkLat) &&
        Number.isFinite(numJerkLongActions) &&
        Number.isFinite(numJerkLatActions) &&
        numJerkLongActions > 0 &&
        numJerkLatActions > 0

    if (!valid) return getJerkPncActionSpace()

    const jerkLong = ensureZeroInAxisSorted(linspace(minJerkLong, maxJerkLong, numJerkLongActions))
    // Use nonuniform lateral action space when gamma is provided (mirrors Python _build_nonuniform_lateral_action_space)
    const useGamma = Number.isFinite(gamma) && gamma >= 1.0
    const jerkLat = useGamma
        ? buildNonuniformLateralActionSpace(minJerkLat, maxJerkLat, numJerkLatActions, gamma)
        : ensureZeroInAxisSorted(linspace(minJerkLat, maxJerkLat, numJerkLatActions))

    return {
        jerkLong,
        jerkLat,
        nLong: jerkLong.length,
        nLat: jerkLat.length,
        size: jerkLong.length * jerkLat.length
    }
}

/**
 * Map a flattened action index -> (iLong, iLat)
 *
 * IMPORTANT: Python uses `product(self.jerk_long, self.jerk_lat)`.
 * That means jerk_lat (the second axis) changes fastest.
 */
export function flatTo2D(actionIdx: number, nLat: number): { iLong: number; iLat: number } {
    const iLong = Math.floor(actionIdx / nLat)
    const iLat = actionIdx - iLong * nLat
    return { iLong, iLat }
}

export function twoDToFlat(iLong: number, iLat: number, nLat: number): number {
    return iLong * nLat + iLat
}

/** Convert log-probabilities to probabilities in a numerically-stable way. */
export function logProbsToProbs(logProbs: number[]): number[] {
    if (!logProbs?.length) return []
    let max = -Infinity
    for (const lp of logProbs) if (Number.isFinite(lp) && lp > max) max = lp
    if (!Number.isFinite(max)) return logProbs.map(() => 0)

    const exps = logProbs.map((lp) => (Number.isFinite(lp) ? Math.exp(lp - max) : 0))
    const sum = exps.reduce((a, b) => a + b, 0)
    if (sum <= 0 || !Number.isFinite(sum)) return exps.map(() => 0)
    return exps.map((v) => v / sum)
}

/** Reshape a flat vector of length nLong*nLat into [nLong][nLat]. */
export function reshape2D<T>(flat: T[], nLong: number, nLat: number, fill: T): T[][] {
    const grid: T[][] = Array.from({ length: nLong }, () => Array.from({ length: nLat }, () => fill))
    const N = Math.min(flat.length, nLong * nLat)
    for (let k = 0; k < N; k++) {
        const iLong = Math.floor(k / nLat)
        const iLat = k - iLong * nLat
        if (iLong < 0 || iLat < 0 || iLong >= nLong || iLat >= nLat) continue
        const v = flat[k]
        if (v !== undefined) grid[iLong]![iLat] = v
    }
    return grid
}

// ─── Wheel-rate distribution from policy logprobs ──────────────────────────

/** STEER_RATIO: steering wheel angle / tire angle (from vehicle_constants.py). */
const STEER_RATIO = 12.6

/** Steering angle limits in radians (from constants.py). */
const MIN_STEERING_ANGLE = (-432 * Math.PI) / 180 / 12.6 // ≈ -0.5988 rad
const MAX_STEERING_ANGLE = (432 * Math.PI) / 180 / 12.6 // ≈ +0.5988 rad

/** Default dynamics constants (from jerk_pnc_model.py / constants.py). */
const DEFAULTS = {
    lateralModeTransitionSpeed: 5.0,
    lateralModeTransitionWidth: 0.4,
    maxTireAngleRateLowerBound: 0.03,
    maxTireAngleRateUpperBound: 0.3 * 0.7,
    maxTireAngleRateGain: 1.39 * 0.7,
    minALat: -3.0,
    maxALat: 3.0,
    minALong: -3.0,
    maxALong: 1.5,
    minSpeed: 0.0,
    maxSpeed: 40.0,
    frameTimeInterval: 0.1
}

export type SteeringRateDynamicsConfig = {
    lateralModeTransitionSpeed?: number
    lateralModeTransitionWidth?: number
    lateralActionSpaceShapeGamma?: number
    maxTireAngleRateLowerBound?: number
    maxTireAngleRateUpperBound?: number
    maxTireAngleRateGain?: number
    minALat?: number
    maxALat?: number
    minALong?: number
    maxALong?: number
    frameTimeInterval?: number
}

export type AgentDynamicsState = {
    speed: number
    vLong: number
    steering: number
    acceleration: number
    wheelbase: number
}

function sigmoid(x: number): number {
    return 1.0 / (1.0 + Math.exp(-x))
}

function clamp(v: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, v))
}

/**
 * Build the nonuniform lateral action space (mirrors Python
 * `_build_nonuniform_lateral_action_space`).
 */
export function buildNonuniformLateralActionSpace(minJerkLat: number, maxJerkLat: number, numSteps: number, gamma: number): number[] {
    if (numSteps <= 0) return [0]
    const u = linspace(-1.0, 1.0, numSteps)
    const shaped = u.map((v) => Math.sign(v) * Math.pow(Math.abs(v), gamma))
    const jerkLat = shaped.map((s) => (s >= 0 ? s * maxJerkLat : s * Math.abs(minJerkLat)))
    return ensureZeroInAxisSorted(jerkLat)
}

/**
 * Marginalize the 2D probability grid over jerk_long to get P(jerk_lat).
 */
export function marginalizeLateral(probs: number[], nLong: number, nLat: number): number[] {
    const marginal = new Array(nLat).fill(0)
    for (let iLong = 0; iLong < nLong; iLong++) {
        for (let iLat = 0; iLat < nLat; iLat++) {
            marginal[iLat] += probs[iLong * nLat + iLat] ?? 0
        }
    }
    return marginal
}

/**
 * Marginalize the 2D probability grid over jerk_lat to get P(jerk_long).
 */
export function marginalizeLongitudinal(probs: number[], nLong: number, nLat: number): number[] {
    const marginal = new Array(nLong).fill(0)
    for (let iLong = 0; iLong < nLong; iLong++) {
        for (let iLat = 0; iLat < nLat; iLat++) {
            marginal[iLong] += probs[iLong * nLat + iLat] ?? 0
        }
    }
    return marginal
}

/**
 * For a single lateral action token, compute the INTENDED target wheel rate (°/s)
 * **before** delta-bound and MIN/MAX clamping.
 * targetWheelRate = (targetSteering - currentSteering) / dt * rad2deg * STEER_RATIO
 * Each token maps to a unique value, suitable for distribution visualization.
 */
export function computeTargetWheelRateForLatAction(
    latCommand: number,
    state: AgentDynamicsState,
    actionSpaceConfig: JerkPncActionSpaceConfig,
    dynamicsConfig?: SteeringRateDynamicsConfig,
    jerkLongAction?: number
): number {
    const cfg = { ...DEFAULTS, ...dynamicsConfig }
    const t = cfg.frameTimeInterval
    const { speed, vLong, steering, acceleration, wheelbase } = state
    const latCmdScale = Math.max(Math.abs(actionSpaceConfig.minJerkLat), Math.abs(actionSpaceConfig.maxJerkLat), 1e-3)

    // Branch A (high speed): lateral jerk → target lateral accel → curvature → steering
    const currentCurvature = Math.tan(steering) / wheelbase
    const currentALat = vLong * vLong * currentCurvature
    let targetALat = currentALat + latCommand * t
    targetALat = clamp(targetALat, cfg.minALat, cfg.maxALat)

    const jerkLong = jerkLongAction ?? 0
    let newAccel = acceleration + jerkLong * t
    newAccel = clamp(newAccel, cfg.minALong, cfg.maxALong)
    const predVLong = clamp(vLong + 0.5 * (acceleration + newAccel) * t, cfg.minSpeed, cfg.maxSpeed)
    const targetCurvature = targetALat / Math.max(predVLong * predVLong, 1.0)
    const targetSteeringJerk = Math.atan(targetCurvature * wheelbase)

    // Branch B (low speed): normalized steering-rate command
    // _lat_command_to_steering_rate: latCmdNorm * rateBound
    const rateBound = clamp(cfg.maxTireAngleRateGain / Math.max(speed, 1e-3), cfg.maxTireAngleRateLowerBound, cfg.maxTireAngleRateUpperBound)
    const latCmdNorm = clamp(latCommand / latCmdScale, -1.0, 1.0)
    const steeringRateCmd = latCmdNorm * rateBound
    const targetSteeringRate = steering + steeringRateCmd * t

    // Blend by speed
    const w = sigmoid((speed - cfg.lateralModeTransitionSpeed) / Math.max(cfg.lateralModeTransitionWidth ?? 0.4, 1e-3))
    const targetSteering = w * targetSteeringJerk + (1 - w) * targetSteeringRate

    // Return target rate in °/s (NO delta-bound or min/max clamping)
    return ((targetSteering - steering) / t) * (180 / Math.PI) * STEER_RATIO
}

/**
 * Compute target-wheel-rate distribution: for each lateral token, compute the
 * intended target wheel rate and pair it with log10 of the marginalized probability.
 * Returns sorted by rate. No delta-bound clamping, so each token maps to a unique X.
 */
export function computeTargetWheelRateDistribution(
    probs: number[],
    actionSpace: JerkPncActionSpace,
    state: AgentDynamicsState,
    actionSpaceConfig: JerkPncActionSpaceConfig,
    dynamicsConfig?: SteeringRateDynamicsConfig
): { wheelRates: number[]; logProbs: number[] } {
    const { nLong, nLat, jerkLat, jerkLong } = actionSpace

    const latMarginal = marginalizeLateral(probs, nLong, nLat)

    const longMarginal = marginalizeLongitudinal(probs, nLong, nLat)
    let bestLongIdx = 0
    for (let i = 1; i < longMarginal.length; i++) {
        if ((longMarginal[i] ?? 0) > (longMarginal[bestLongIdx] ?? 0)) bestLongIdx = i
    }
    const bestJerkLong = jerkLong[bestLongIdx] ?? 0

    const pairs: [number, number][] = []
    for (let iLat = 0; iLat < nLat; iLat++) {
        const wr = computeTargetWheelRateForLatAction(jerkLat[iLat] ?? 0, state, actionSpaceConfig, dynamicsConfig, bestJerkLong)
        const p = latMarginal[iLat] ?? 0
        pairs.push([wr, p > 0 ? Math.log10(p) : -30])
    }
    pairs.sort((a, b) => a[0] - b[0])

    return {
        wheelRates: pairs.map((p) => p[0]),
        logProbs: pairs.map((p) => p[1])
    }
}

/** Max wheel rate in deg/s for chart axis range.
 *  Uses maxTireAngleRateUpperBound (0.21 rad/s) as the physical limit,
 *  with 2x margin to accommodate unclamped target rates.
 */
export function getMaxWheelRate(): number {
    const rateUpperBound = DEFAULTS.maxTireAngleRateUpperBound // 0.21 rad/s
    return rateUpperBound * (180 / Math.PI) * STEER_RATIO
    // ≈ 0.21 * 57.3 * 12.6 ≈ 151.5 deg/s
}
