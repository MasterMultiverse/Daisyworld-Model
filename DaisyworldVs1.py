#Global temperature now depends on planetary mean albedo, rather than treating each cell as an independent planet.
#Local black and white surface temperatures are generated from their albedo difference relative to the planetary mean.
#Only daisies that survive the death stage can reproduce.
#Black and white daisies use different establishment temperatures and growth rates.
#Temperature, albedo, populations, and snapshots now describe the same simulation state.
#The lifeless control uses the same thermal relaxation timescale as the living planet.
#Climate spin-up prevents the initial map from being artificially extreme.
#The final biological update is skipped because it would otherwise create an unrecorded state with no corresponding climate calculation.



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ============================================================
# 2D SPATIAL DAISYWORLD WITH HEAT DIFFUSION
# ============================================================
#
# Cell values:
#   0 = bare ground
#   1 = black daisy
#   2 = white daisy
#
# Model sequence during each simulation step:
#
#   1. Calculate solar luminosity.
#   2. Calculate planetary mean albedo.
#   3. Calculate global radiative equilibrium temperature.
#   4. Calculate local target temperatures from albedo differences.
#   5. Relax and diffuse the spatial temperature field.
#   6. Record climate and population values for the current state.
#   7. Apply daisy death and reproduction.
#
# Required packages:
#   numpy
#   matplotlib
#
# ============================================================


# ============================================================
# USER SETTINGS
# ============================================================

# Size of the square lattice
GRID_SIZE = 80

# Number of simulation steps
STEPS = 1400

# Starting and ending solar luminosity
LUMINOSITY_START = 0.65
LUMINOSITY_END = 1.45

# Initial surface fractions
INITIAL_BLACK_FRACTION = 0.05
INITIAL_WHITE_FRACTION = 0.05

# Surface albedos
ALBEDO_BLACK = 0.25
ALBEDO_GROUND = 0.50
ALBEDO_WHITE = 0.75

# Biological parameters
OPTIMUM_TEMPERATURE = 22.5
GROWTH_COEFFICIENT = 0.003265
DEATH_PROBABILITY = 0.015

# Local colonization strength
COLONIZATION_PROBABILITY = 0.35

# Small long-distance seed dispersal probability
#
# Set this to 0.0 for strictly local reproduction.
# A small nonzero value allows recolonization after extinction.
LONG_DISTANCE_SEEDING_PROBABILITY = 0.00005

# Climate parameters
TEMPERATURE_RELAXATION = 0.12
HEAT_DIFFUSION = 0.15

# Number of climate substeps during each biological step
CLIMATE_ITERATIONS_PER_STEP = 4

# Local temperature effect produced by albedo differences
LOCAL_HEAT_FACTOR = 20.0

# Small random local temperature fluctuations
TEMPERATURE_NOISE = 0.05

# Number of initial climate-only spin-up steps
CLIMATE_SPINUP_STEPS = 100

# Number of simulation steps between progress messages
PROGRESS_INTERVAL = 100

# Random seed
#
# Change this value to produce a different stochastic simulation.
RANDOM_SEED = 42


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

# Stefan-Boltzmann constant in W m^-2 K^-4
STEFAN_BOLTZMANN = 5.670374419e-8

# Effective Daisyworld solar flux in W/m^2
SOLAR_FLUX = 917.0


# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def validate_parameters():
    """
    Check model settings before starting the simulation.
    """

    if GRID_SIZE < 2:
        raise ValueError(
            "GRID_SIZE must be at least 2."
        )

    if STEPS < 2:
        raise ValueError(
            "STEPS must be at least 2."
        )

    if CLIMATE_ITERATIONS_PER_STEP < 1:
        raise ValueError(
            "CLIMATE_ITERATIONS_PER_STEP must be at least 1."
        )

    if CLIMATE_SPINUP_STEPS < 0:
        raise ValueError(
            "CLIMATE_SPINUP_STEPS cannot be negative."
        )

    if not 0.0 <= INITIAL_BLACK_FRACTION <= 1.0:
        raise ValueError(
            "INITIAL_BLACK_FRACTION must be between 0 and 1."
        )

    if not 0.0 <= INITIAL_WHITE_FRACTION <= 1.0:
        raise ValueError(
            "INITIAL_WHITE_FRACTION must be between 0 and 1."
        )

    if (
        INITIAL_BLACK_FRACTION
        + INITIAL_WHITE_FRACTION
        > 1.0
    ):
        raise ValueError(
            "Initial black and white daisy fractions "
            "cannot add up to more than 1."
        )

    if not 0.0 <= DEATH_PROBABILITY <= 1.0:
        raise ValueError(
            "DEATH_PROBABILITY must be between 0 and 1."
        )

    if not 0.0 <= COLONIZATION_PROBABILITY <= 1.0:
        raise ValueError(
            "COLONIZATION_PROBABILITY must be between 0 and 1."
        )

    if not 0.0 <= LONG_DISTANCE_SEEDING_PROBABILITY <= 1.0:
        raise ValueError(
            "LONG_DISTANCE_SEEDING_PROBABILITY must be "
            "between 0 and 1."
        )

    if TEMPERATURE_RELAXATION < 0.0:
        raise ValueError(
            "TEMPERATURE_RELAXATION cannot be negative."
        )

    if HEAT_DIFFUSION < 0.0:
        raise ValueError(
            "HEAT_DIFFUSION cannot be negative."
        )

    if TEMPERATURE_NOISE < 0.0:
        raise ValueError(
            "TEMPERATURE_NOISE cannot be negative."
        )

    # Conservative stability check for the combined explicit
    # diffusion and relaxation calculation.
    stability_number = (
        4.0 * HEAT_DIFFUSION
        + TEMPERATURE_RELAXATION
    )

    if stability_number > 1.0:
        raise ValueError(
            "The climate calculation may be unstable because "
            "4 * HEAT_DIFFUSION + TEMPERATURE_RELAXATION "
            "is greater than 1. Reduce HEAT_DIFFUSION or "
            "TEMPERATURE_RELAXATION."
        )

    albedos = [
        ALBEDO_BLACK,
        ALBEDO_GROUND,
        ALBEDO_WHITE
    ]

    for albedo in albedos:
        if not 0.0 <= albedo < 1.0:
            raise ValueError(
                "Every albedo must be at least 0 and less than 1."
            )

    if not (
        ALBEDO_BLACK
        < ALBEDO_GROUND
        < ALBEDO_WHITE
    ):
        raise ValueError(
            "The expected albedo order is "
            "ALBEDO_BLACK < ALBEDO_GROUND < ALBEDO_WHITE."
        )


# ============================================================
# LUMINOSITY
# ============================================================

def luminosity_at_step(step):
    """
    Return linearly increasing solar luminosity.
    """

    fraction = step / max(STEPS - 1, 1)

    return (
        LUMINOSITY_START
        + fraction
        * (LUMINOSITY_END - LUMINOSITY_START)
    )


# ============================================================
# WORLD AND ALBEDO FUNCTIONS
# ============================================================

def initialize_world():
    """
    Create the initial random distribution of daisies.
    """

    world = np.zeros(
        (GRID_SIZE, GRID_SIZE),
        dtype=np.int8
    )

    random_values = rng.random(
        world.shape
    )

    black_limit = INITIAL_BLACK_FRACTION

    white_limit = (
        INITIAL_BLACK_FRACTION
        + INITIAL_WHITE_FRACTION
    )

    world[
        random_values < black_limit
    ] = 1

    world[
        (random_values >= black_limit)
        & (random_values < white_limit)
    ] = 2

    return world


def make_albedo_map(world):
    """
    Convert the biological lattice into a map of albedos.
    """

    albedo_map = np.full(
        world.shape,
        ALBEDO_GROUND,
        dtype=float
    )

    albedo_map[world == 1] = ALBEDO_BLACK
    albedo_map[world == 2] = ALBEDO_WHITE

    return albedo_map


# ============================================================
# TEMPERATURE FUNCTIONS
# ============================================================

def radiative_equilibrium_temperature(
    luminosity,
    planetary_albedo
):
    """
    Calculate global radiative equilibrium temperature.

    Energy balance:

        absorbed solar radiation = emitted thermal radiation

        S * L * (1 - A) = sigma * T^4

    Temperature is returned in degrees Celsius.
    """

    absorbed_energy = (
        SOLAR_FLUX
        * luminosity
        * (1.0 - planetary_albedo)
    )

    absorbed_energy = max(
        float(absorbed_energy),
        1.0e-12
    )

    temperature_kelvin = (
        absorbed_energy
        / STEFAN_BOLTZMANN
    ) ** 0.25

    return temperature_kelvin - 273.15


def calculate_local_target_temperature(
    luminosity,
    albedo_map
):
    """
    Calculate the target temperature for every lattice cell.

    First, the global planetary temperature is calculated using
    mean planetary albedo.

    Local temperatures are then adjusted according to the
    difference between planetary mean albedo and local albedo.

    Black surfaces become warmer than the planetary average.
    White surfaces become cooler than the planetary average.
    """

    mean_planetary_albedo = float(
        np.mean(albedo_map)
    )

    global_temperature = (
        radiative_equilibrium_temperature(
            luminosity,
            mean_planetary_albedo
        )
    )

    local_target_temperature = (
        global_temperature
        + LOCAL_HEAT_FACTOR
        * (
            mean_planetary_albedo
            - albedo_map
        )
    )

    return (
        local_target_temperature,
        global_temperature,
        mean_planetary_albedo
    )


def calculate_laplacian(field):
    """
    Calculate the discrete two-dimensional Laplacian.

    Periodic boundary conditions are used. A cell at one edge is
    connected to the corresponding cell on the opposite edge.
    """

    north = np.roll(
        field,
        1,
        axis=0
    )

    south = np.roll(
        field,
        -1,
        axis=0
    )

    west = np.roll(
        field,
        1,
        axis=1
    )

    east = np.roll(
        field,
        -1,
        axis=1
    )

    return (
        north
        + south
        + west
        + east
        - 4.0 * field
    )


def update_temperature(
    temperature,
    target_temperature,
    include_noise=True
):
    """
    Update the spatial temperature field.

    The calculation includes:

        1. Relaxation toward local target temperature
        2. Diffusion between neighboring cells
        3. Optional random local thermal fluctuations
    """

    updated_temperature = temperature.copy()

    for _ in range(CLIMATE_ITERATIONS_PER_STEP):

        laplacian = calculate_laplacian(
            updated_temperature
        )

        radiative_change = (
            TEMPERATURE_RELAXATION
            * (
                target_temperature
                - updated_temperature
            )
        )

        diffusion_change = (
            HEAT_DIFFUSION
            * laplacian
        )

        updated_temperature = (
            updated_temperature
            + radiative_change
            + diffusion_change
        )

    if include_noise and TEMPERATURE_NOISE > 0.0:

        updated_temperature += rng.normal(
            loc=0.0,
            scale=TEMPERATURE_NOISE,
            size=updated_temperature.shape
        )

    if not np.all(
        np.isfinite(updated_temperature)
    ):
        raise RuntimeError(
            "The temperature field became numerically unstable. "
            "Reduce HEAT_DIFFUSION or TEMPERATURE_RELAXATION."
        )

    return updated_temperature


def update_lifeless_temperature(
    lifeless_temperature,
    luminosity
):
    """
    Update the lifeless planet using the same thermal-relaxation
    timescale as the living planet.

    The lifeless control has only bare-ground albedo and does not
    include spatial noise.
    """

    target_temperature = (
        radiative_equilibrium_temperature(
            luminosity,
            ALBEDO_GROUND
        )
    )

    updated_temperature = (
        lifeless_temperature
    )

    for _ in range(CLIMATE_ITERATIONS_PER_STEP):

        updated_temperature += (
            TEMPERATURE_RELAXATION
            * (
                target_temperature
                - updated_temperature
            )
        )

    return updated_temperature


# ============================================================
# BIOLOGICAL FUNCTIONS
# ============================================================

def growth_rate(temperature):
    """
    Calculate daisy growth suitability.

    Maximum growth occurs at OPTIMUM_TEMPERATURE.

    Growth falls parabolically with distance from the optimum
    and is clipped to the range from 0 to 1.
    """

    growth = (
        1.0
        - GROWTH_COEFFICIENT
        * (
            OPTIMUM_TEMPERATURE
            - temperature
        ) ** 2
    )

    return np.clip(
        growth,
        0.0,
        1.0
    )


def neighbor_fraction(mask):
    """
    Calculate the fraction of the eight neighboring cells that
    satisfy the supplied Boolean mask.
    """

    northwest = np.roll(
        np.roll(mask, 1, axis=0),
        1,
        axis=1
    )

    north = np.roll(
        mask,
        1,
        axis=0
    )

    northeast = np.roll(
        np.roll(mask, 1, axis=0),
        -1,
        axis=1
    )

    west = np.roll(
        mask,
        1,
        axis=1
    )

    east = np.roll(
        mask,
        -1,
        axis=1
    )

    southwest = np.roll(
        np.roll(mask, -1, axis=0),
        1,
        axis=1
    )

    south = np.roll(
        mask,
        -1,
        axis=0
    )

    southeast = np.roll(
        np.roll(mask, -1, axis=0),
        -1,
        axis=1
    )

    neighbor_count = (
        northwest.astype(float)
        + north.astype(float)
        + northeast.astype(float)
        + west.astype(float)
        + east.astype(float)
        + southwest.astype(float)
        + south.astype(float)
        + southeast.astype(float)
    )

    return neighbor_count / 8.0


def calculate_establishment_temperatures(
    temperature
):
    """
    Estimate the temperature an empty ground cell would have if
    it were colonized by a black or white daisy.

    The current empty cell is treated as ground. Replacing ground
    with black vegetation warms the cell. Replacing ground with
    white vegetation cools the cell.
    """

    black_temperature_change = (
        LOCAL_HEAT_FACTOR
        * (
            ALBEDO_GROUND
            - ALBEDO_BLACK
        )
    )

    white_temperature_change = (
        LOCAL_HEAT_FACTOR
        * (
            ALBEDO_GROUND
            - ALBEDO_WHITE
        )
    )

    black_establishment_temperature = (
        temperature
        + black_temperature_change
    )

    white_establishment_temperature = (
        temperature
        + white_temperature_change
    )

    return (
        black_establishment_temperature,
        white_establishment_temperature
    )


def reproduce_daisies(
    world,
    temperature
):
    """
    Perform one biological update.

    Biological sequence:

        1. Existing daisies may die.
        2. Surviving daisies act as reproductive sources.
        3. Empty cells may be colonized.
        4. Species-specific establishment temperatures determine
           growth suitability.
        5. Competition is resolved if both species successfully
           attempt to colonize the same cell.
    """

    updated_world = world.copy()

    # --------------------------------------------------------
    # Death
    # --------------------------------------------------------

    black_mask_before_death = (
        updated_world == 1
    )

    white_mask_before_death = (
        updated_world == 2
    )

    black_death = (
        black_mask_before_death
        & (
            rng.random(updated_world.shape)
            < DEATH_PROBABILITY
        )
    )

    white_death = (
        white_mask_before_death
        & (
            rng.random(updated_world.shape)
            < DEATH_PROBABILITY
        )
    )

    updated_world[black_death] = 0
    updated_world[white_death] = 0

    # Only surviving daisies may reproduce.
    black_survivors = (
        updated_world == 1
    )

    white_survivors = (
        updated_world == 2
    )

    empty_mask = (
        updated_world == 0
    )

    # --------------------------------------------------------
    # Surviving reproductive neighbors
    # --------------------------------------------------------

    black_neighbors = neighbor_fraction(
        black_survivors
    )

    white_neighbors = neighbor_fraction(
        white_survivors
    )

    # --------------------------------------------------------
    # Species-specific local growth conditions
    # --------------------------------------------------------

    (
        black_establishment_temperature,
        white_establishment_temperature
    ) = calculate_establishment_temperatures(
        temperature
    )

    black_growth = growth_rate(
        black_establishment_temperature
    )

    white_growth = growth_rate(
        white_establishment_temperature
    )

    # --------------------------------------------------------
    # Colonization probabilities
    # --------------------------------------------------------

    black_probability = (
        COLONIZATION_PROBABILITY
        * black_neighbors
        * black_growth
    )

    white_probability = (
        COLONIZATION_PROBABILITY
        * white_neighbors
        * white_growth
    )

    # Small long-distance seed dispersal.
    black_probability += (
        LONG_DISTANCE_SEEDING_PROBABILITY
        * black_growth
    )

    white_probability += (
        LONG_DISTANCE_SEEDING_PROBABILITY
        * white_growth
    )

    # Colonization is possible only on empty ground.
    black_probability[~empty_mask] = 0.0
    white_probability[~empty_mask] = 0.0

    black_probability = np.clip(
        black_probability,
        0.0,
        1.0
    )

    white_probability = np.clip(
        white_probability,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Colonization attempts
    # --------------------------------------------------------

    black_success = (
        rng.random(updated_world.shape)
        < black_probability
    )

    white_success = (
        rng.random(updated_world.shape)
        < white_probability
    )

    black_success &= empty_mask
    white_success &= empty_mask

    black_only = (
        black_success
        & ~white_success
    )

    white_only = (
        white_success
        & ~black_success
    )

    updated_world[black_only] = 1
    updated_world[white_only] = 2

    # --------------------------------------------------------
    # Resolve direct competition
    # --------------------------------------------------------

    competition = (
        black_success
        & white_success
    )

    if np.any(competition):

        black_strength = (
            black_probability[competition]
        )

        white_strength = (
            white_probability[competition]
        )

        total_strength = (
            black_strength
            + white_strength
        )

        probability_black_wins = np.full(
            black_strength.shape,
            0.5,
            dtype=float
        )

        np.divide(
            black_strength,
            total_strength,
            out=probability_black_wins,
            where=total_strength > 0.0
        )

        probability_black_wins = np.clip(
            probability_black_wins,
            0.0,
            1.0
        )

        black_wins = (
            rng.random(
                probability_black_wins.size
            )
            < probability_black_wins
        )

        competition_rows, competition_columns = (
            np.where(competition)
        )

        updated_world[
            competition_rows[black_wins],
            competition_columns[black_wins]
        ] = 1

        updated_world[
            competition_rows[~black_wins],
            competition_columns[~black_wins]
        ] = 2

    return updated_world


# ============================================================
# INITIALIZE MODEL
# ============================================================

validate_parameters()

print("=" * 72)
print("2D DAISYWORLD WITH HEAT DIFFUSION")
print("=" * 72)

print(
    f"Grid size                  : "
    f"{GRID_SIZE} x {GRID_SIZE}"
)

print(
    f"Simulation steps           : "
    f"{STEPS}"
)

print(
    f"Solar luminosity           : "
    f"{LUMINOSITY_START:.2f} to "
    f"{LUMINOSITY_END:.2f}"
)

print(
    f"Temperature relaxation     : "
    f"{TEMPERATURE_RELAXATION}"
)

print(
    f"Heat diffusion             : "
    f"{HEAT_DIFFUSION}"
)

print(
    f"Local heat factor          : "
    f"{LOCAL_HEAT_FACTOR}"
)

print(
    f"Climate spin-up steps      : "
    f"{CLIMATE_SPINUP_STEPS}"
)

print(
    f"Random seed                : "
    f"{RANDOM_SEED}"
)

print("=" * 72)

world = initialize_world()

initial_albedo_map = make_albedo_map(
    world
)

initial_mean_albedo = float(
    np.mean(initial_albedo_map)
)

initial_planetary_temperature = (
    radiative_equilibrium_temperature(
        LUMINOSITY_START,
        initial_mean_albedo
    )
)

# Begin with a uniform planetary temperature. The spin-up phase
# develops the local temperature structure consistently.
temperature = np.full(
    world.shape,
    initial_planetary_temperature,
    dtype=float
)

lifeless_temperature = (
    radiative_equilibrium_temperature(
        LUMINOSITY_START,
        ALBEDO_GROUND
    )
)


# ============================================================
# CLIMATE SPIN-UP
# ============================================================

print("\nClimate spin-up started...")

for spinup_step in range(CLIMATE_SPINUP_STEPS):

    spinup_albedo_map = make_albedo_map(
        world
    )

    (
        spinup_target_temperature,
        _,
        _
    ) = calculate_local_target_temperature(
        LUMINOSITY_START,
        spinup_albedo_map
    )

    temperature = update_temperature(
        temperature,
        spinup_target_temperature,
        include_noise=False
    )

    lifeless_temperature = (
        update_lifeless_temperature(
            lifeless_temperature,
            LUMINOSITY_START
        )
    )

print("Climate spin-up completed.")


# ============================================================
# HISTORY ARRAYS
# ============================================================

time_history = np.arange(
    STEPS
)

luminosity_history = np.zeros(
    STEPS,
    dtype=float
)

black_history = np.zeros(
    STEPS,
    dtype=float
)

white_history = np.zeros(
    STEPS,
    dtype=float
)

bare_history = np.zeros(
    STEPS,
    dtype=float
)

mean_temperature_history = np.zeros(
    STEPS,
    dtype=float
)

global_radiative_temperature_history = np.zeros(
    STEPS,
    dtype=float
)

minimum_temperature_history = np.zeros(
    STEPS,
    dtype=float
)

maximum_temperature_history = np.zeros(
    STEPS,
    dtype=float
)

planetary_albedo_history = np.zeros(
    STEPS,
    dtype=float
)

lifeless_temperature_history = np.zeros(
    STEPS,
    dtype=float
)


# ============================================================
# SNAPSHOT SETTINGS
# ============================================================

middle_step = STEPS // 2

initial_world_snapshot = None
initial_temperature_snapshot = None
initial_luminosity_snapshot = None

middle_world_snapshot = None
middle_temperature_snapshot = None
middle_luminosity_snapshot = None

final_world_snapshot = None
final_temperature_snapshot = None
final_luminosity_snapshot = None


# ============================================================
# MAIN SIMULATION
# ============================================================

print("\nMain simulation started...\n")

for step in range(STEPS):

    luminosity = luminosity_at_step(
        step
    )

    # --------------------------------------------------------
    # Climate responds to the current biological world.
    # --------------------------------------------------------

    albedo_map = make_albedo_map(
        world
    )

    (
        local_target_temperature,
        global_radiative_temperature,
        mean_planetary_albedo
    ) = calculate_local_target_temperature(
        luminosity,
        albedo_map
    )

    temperature = update_temperature(
        temperature,
        local_target_temperature,
        include_noise=True
    )

    lifeless_temperature = (
        update_lifeless_temperature(
            lifeless_temperature,
            luminosity
        )
    )

    # --------------------------------------------------------
    # Record the current synchronized world and climate state.
    # --------------------------------------------------------

    black_fraction = float(
        np.mean(world == 1)
    )

    white_fraction = float(
        np.mean(world == 2)
    )

    bare_fraction = float(
        np.mean(world == 0)
    )

    mean_temperature = float(
        np.mean(temperature)
    )

    minimum_temperature = float(
        np.min(temperature)
    )

    maximum_temperature = float(
        np.max(temperature)
    )

    luminosity_history[step] = (
        luminosity
    )

    black_history[step] = (
        black_fraction
    )

    white_history[step] = (
        white_fraction
    )

    bare_history[step] = (
        bare_fraction
    )

    mean_temperature_history[step] = (
        mean_temperature
    )

    global_radiative_temperature_history[step] = (
        global_radiative_temperature
    )

    minimum_temperature_history[step] = (
        minimum_temperature
    )

    maximum_temperature_history[step] = (
        maximum_temperature
    )

    planetary_albedo_history[step] = (
        mean_planetary_albedo
    )

    lifeless_temperature_history[step] = (
        lifeless_temperature
    )

    # --------------------------------------------------------
    # Save synchronized spatial snapshots.
    # --------------------------------------------------------

    if step == 0:

        initial_world_snapshot = (
            world.copy()
        )

        initial_temperature_snapshot = (
            temperature.copy()
        )

        initial_luminosity_snapshot = (
            luminosity
        )

    if step == middle_step:

        middle_world_snapshot = (
            world.copy()
        )

        middle_temperature_snapshot = (
            temperature.copy()
        )

        middle_luminosity_snapshot = (
            luminosity
        )

    if step == STEPS - 1:

        final_world_snapshot = (
            world.copy()
        )

        final_temperature_snapshot = (
            temperature.copy()
        )

        final_luminosity_snapshot = (
            luminosity
        )

    # --------------------------------------------------------
    # Console progress.
    # --------------------------------------------------------

    if (
        step % PROGRESS_INTERVAL == 0
        or step == STEPS - 1
    ):

        total_daisies = (
            black_fraction
            + white_fraction
        )

        print(
            f"Step {step + 1:4d}/{STEPS} | "
            f"L = {luminosity:.3f} | "
            f"T = {mean_temperature:7.2f} C | "
            f"Black = {black_fraction:6.3f} | "
            f"White = {white_fraction:6.3f} | "
            f"Total = {total_daisies:6.3f}"
        )

    # --------------------------------------------------------
    # Biology changes at the end of the time step.
    #
    # The final recorded snapshot is not followed by another
    # biological update because that unrecorded state would not
    # have a corresponding climate value.
    # --------------------------------------------------------

    if step < STEPS - 1:

        world = reproduce_daisies(
            world,
            temperature
        )


# ============================================================
# RESULT CHECKS
# ============================================================

coverage_sum = (
    black_history
    + white_history
    + bare_history
)

if not np.allclose(
    coverage_sum,
    1.0,
    atol=1.0e-10
):
    raise RuntimeError(
        "Recorded surface fractions do not add up to 1."
    )

if not np.all(
    np.isfinite(mean_temperature_history)
):
    raise RuntimeError(
        "Invalid values were found in temperature history."
    )

if not np.all(
    np.isfinite(lifeless_temperature_history)
):
    raise RuntimeError(
        "Invalid values were found in the lifeless control."
    )


# ============================================================
# SUMMARY
# ============================================================

total_daisy_history = (
    black_history
    + white_history
)

maximum_daisy_index = int(
    np.argmax(total_daisy_history)
)

maximum_daisy_coverage = float(
    total_daisy_history[
        maximum_daisy_index
    ]
)

luminosity_at_maximum_daisy_coverage = float(
    luminosity_history[
        maximum_daisy_index
    ]
)

living_minus_lifeless = (
    mean_temperature_history
    - lifeless_temperature_history
)

largest_temperature_difference_index = int(
    np.argmax(
        np.abs(living_minus_lifeless)
    )
)

largest_temperature_difference = float(
    living_minus_lifeless[
        largest_temperature_difference_index
    ]
)

luminosity_at_largest_difference = float(
    luminosity_history[
        largest_temperature_difference_index
    ]
)

print("\nSimulation completed.")
print("=" * 72)
print("FINAL RESULTS")
print("=" * 72)

print(
    f"Final luminosity            : "
    f"{luminosity_history[-1]:.3f}"
)

print(
    f"Final mean temperature      : "
    f"{mean_temperature_history[-1]:.2f} C"
)

print(
    f"Final lifeless temperature  : "
    f"{lifeless_temperature_history[-1]:.2f} C"
)

print(
    f"Final black coverage        : "
    f"{black_history[-1]:.3f}"
)

print(
    f"Final white coverage        : "
    f"{white_history[-1]:.3f}"
)

print(
    f"Final bare coverage         : "
    f"{bare_history[-1]:.3f}"
)

print(
    f"Maximum daisy coverage      : "
    f"{maximum_daisy_coverage:.3f}"
)

print(
    f"Luminosity at maximum       : "
    f"{luminosity_at_maximum_daisy_coverage:.3f}"
)

print(
    f"Largest living-control diff : "
    f"{largest_temperature_difference:+.2f} C"
)

print(
    f"Luminosity at largest diff  : "
    f"{luminosity_at_largest_difference:.3f}"
)

print("=" * 72)


# ============================================================
# MAIN RESULTS FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


# ------------------------------------------------------------
# Plot 1: Solar luminosity
# ------------------------------------------------------------

axes[0, 0].plot(
    time_history,
    luminosity_history,
    color="darkorange",
    linewidth=2.5
)

axes[0, 0].set_title(
    "Slowly Increasing Solar Luminosity"
)

axes[0, 0].set_xlabel(
    "Simulation step"
)

axes[0, 0].set_ylabel(
    "Relative luminosity"
)

axes[0, 0].grid(True)


# ------------------------------------------------------------
# Plot 2: Daisy populations
# ------------------------------------------------------------

axes[0, 1].plot(
    luminosity_history,
    black_history,
    color="black",
    linewidth=2.0,
    label="Black daisies"
)

axes[0, 1].plot(
    luminosity_history,
    white_history,
    color="royalblue",
    linewidth=2.0,
    label="White daisies"
)

axes[0, 1].plot(
    luminosity_history,
    bare_history,
    color="saddlebrown",
    linewidth=1.5,
    linestyle="--",
    label="Bare ground"
)

axes[0, 1].set_title(
    "Surface Coverage vs Solar Luminosity"
)

axes[0, 1].set_xlabel(
    "Solar luminosity"
)

axes[0, 1].set_ylabel(
    "Coverage fraction"
)

axes[0, 1].set_ylim(
    0.0,
    1.0
)

axes[0, 1].legend()
axes[0, 1].grid(True)


# ------------------------------------------------------------
# Plot 3: Living and lifeless temperatures
# ------------------------------------------------------------

axes[1, 0].plot(
    luminosity_history,
    mean_temperature_history,
    color="red",
    linewidth=2.5,
    label="Living Daisyworld"
)

axes[1, 0].plot(
    luminosity_history,
    lifeless_temperature_history,
    color="gray",
    linewidth=2.0,
    linestyle="--",
    label="Lifeless control"
)

axes[1, 0].plot(
    luminosity_history,
    global_radiative_temperature_history,
    color="darkorange",
    linewidth=1.3,
    linestyle="-.",
    label="Living-world radiative target"
)

axes[1, 0].axhline(
    OPTIMUM_TEMPERATURE,
    color="green",
    linewidth=1.5,
    linestyle=":",
    label="Optimal growth temperature"
)

axes[1, 0].fill_between(
    luminosity_history,
    minimum_temperature_history,
    maximum_temperature_history,
    color="red",
    alpha=0.12,
    label="Spatial temperature range"
)

axes[1, 0].set_title(
    "Planetary Temperature Regulation"
)

axes[1, 0].set_xlabel(
    "Solar luminosity"
)

axes[1, 0].set_ylabel(
    "Temperature (C)"
)

axes[1, 0].legend(
    fontsize=8
)

axes[1, 0].grid(True)


# ------------------------------------------------------------
# Plot 4: Planetary albedo
# ------------------------------------------------------------

axes[1, 1].plot(
    luminosity_history,
    planetary_albedo_history,
    color="purple",
    linewidth=2.5,
    label="Living planetary albedo"
)

axes[1, 1].axhline(
    ALBEDO_GROUND,
    color="saddlebrown",
    linestyle="--",
    linewidth=1.5,
    label="Bare-ground albedo"
)

axes[1, 1].set_title(
    "Planetary Albedo Feedback"
)

axes[1, 1].set_xlabel(
    "Solar luminosity"
)

axes[1, 1].set_ylabel(
    "Mean planetary albedo"
)

axes[1, 1].legend()
axes[1, 1].grid(True)

fig.suptitle(
    "2D Daisyworld: Biosphere-Climate Feedback",
    fontsize=16
)

fig.tight_layout(
    rect=(0.0, 0.0, 1.0, 0.96)
)


# ============================================================
# SPATIAL MAP FIGURE
# ============================================================

world_colors = [
    "#9b7653",
    "#111111",
    "#f2f2f2"
]

world_cmap = ListedColormap(
    world_colors
)

fig_maps, map_axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)


# ------------------------------------------------------------
# Initial biological world
# ------------------------------------------------------------

map_axes[0, 0].imshow(
    initial_world_snapshot,
    cmap=world_cmap,
    vmin=0,
    vmax=2,
    interpolation="nearest"
)

map_axes[0, 0].set_title(
    f"Initial World\n"
    f"L = {initial_luminosity_snapshot:.2f}"
)

map_axes[0, 0].set_xticks([])
map_axes[0, 0].set_yticks([])


# ------------------------------------------------------------
# Middle biological world
# ------------------------------------------------------------

map_axes[0, 1].imshow(
    middle_world_snapshot,
    cmap=world_cmap,
    vmin=0,
    vmax=2,
    interpolation="nearest"
)

map_axes[0, 1].set_title(
    f"Middle World\n"
    f"L = {middle_luminosity_snapshot:.2f}"
)

map_axes[0, 1].set_xticks([])
map_axes[0, 1].set_yticks([])


# ------------------------------------------------------------
# Final biological world
# ------------------------------------------------------------

map_axes[0, 2].imshow(
    final_world_snapshot,
    cmap=world_cmap,
    vmin=0,
    vmax=2,
    interpolation="nearest"
)

map_axes[0, 2].set_title(
    f"Final World\n"
    f"L = {final_luminosity_snapshot:.2f}"
)

map_axes[0, 2].set_xticks([])
map_axes[0, 2].set_yticks([])


# ------------------------------------------------------------
# Common temperature scale
# ------------------------------------------------------------

temperature_minimum = min(
    float(np.min(initial_temperature_snapshot)),
    float(np.min(middle_temperature_snapshot)),
    float(np.min(final_temperature_snapshot))
)

temperature_maximum = max(
    float(np.max(initial_temperature_snapshot)),
    float(np.max(middle_temperature_snapshot)),
    float(np.max(final_temperature_snapshot))
)


# ------------------------------------------------------------
# Initial temperature map
# ------------------------------------------------------------

temperature_image = map_axes[1, 0].imshow(
    initial_temperature_snapshot,
    cmap="coolwarm",
    vmin=temperature_minimum,
    vmax=temperature_maximum,
    interpolation="nearest"
)

map_axes[1, 0].set_title(
    "Initial Temperature Field"
)

map_axes[1, 0].set_xticks([])
map_axes[1, 0].set_yticks([])


# ------------------------------------------------------------
# Middle temperature map
# ------------------------------------------------------------

map_axes[1, 1].imshow(
    middle_temperature_snapshot,
    cmap="coolwarm",
    vmin=temperature_minimum,
    vmax=temperature_maximum,
    interpolation="nearest"
)

map_axes[1, 1].set_title(
    "Middle Temperature Field"
)

map_axes[1, 1].set_xticks([])
map_axes[1, 1].set_yticks([])


# ------------------------------------------------------------
# Final temperature map
# ------------------------------------------------------------

map_axes[1, 2].imshow(
    final_temperature_snapshot,
    cmap="coolwarm",
    vmin=temperature_minimum,
    vmax=temperature_maximum,
    interpolation="nearest"
)

map_axes[1, 2].set_title(
    "Final Temperature Field"
)

map_axes[1, 2].set_xticks([])
map_axes[1, 2].set_yticks([])


# ------------------------------------------------------------
# Biological map legend
# ------------------------------------------------------------

world_legend = [
    Patch(
        facecolor=world_colors[0],
        edgecolor="black",
        label="Bare ground"
    ),
    Patch(
        facecolor=world_colors[1],
        edgecolor="black",
        label="Black daisies"
    ),
    Patch(
        facecolor=world_colors[2],
        edgecolor="black",
        label="White daisies"
    )
]

fig_maps.legend(
    handles=world_legend,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.94)
)


# ------------------------------------------------------------
# Shared temperature color bar
# ------------------------------------------------------------

colorbar = fig_maps.colorbar(
    temperature_image,
    ax=map_axes[1, :],
    orientation="horizontal",
    fraction=0.06,
    pad=0.10
)

colorbar.set_label(
    "Temperature (C)"
)

fig_maps.suptitle(
    "Spatial Evolution of 2D Daisyworld",
    fontsize=16
)

fig_maps.subplots_adjust(
    top=0.86,
    bottom=0.15,
    wspace=0.10,
    hspace=0.20
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nOpening result graphs...")
print(
    "Close both graph windows to return to the console."
)

plt.show()

input("\nPress Enter to exit the program...")