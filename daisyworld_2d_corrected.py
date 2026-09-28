import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ============================================================
# 2D SPATIAL DAISYWORLD WITH HEAT DIFFUSION
# ============================================================
# Cell values: 0 = bare ground, 1 = black daisy, 2 = white daisy

# ------------------------- USER SETTINGS ----------------------
GRID_SIZE = 80
STEPS = 1400
LUMINOSITY_START = 0.65
LUMINOSITY_END = 1.45
INITIAL_BLACK_FRACTION = 0.05
INITIAL_WHITE_FRACTION = 0.05
ALBEDO_BLACK = 0.25
ALBEDO_GROUND = 0.50
ALBEDO_WHITE = 0.75
OPTIMUM_TEMPERATURE = 22.5
GROWTH_COEFFICIENT = 0.003265
DEATH_PROBABILITY = 0.005
MAX_TEMPERATURE_MORTALITY = 0.10
COLONIZATION_PROBABILITY = 0.70
LONG_DISTANCE_SEEDING_PROBABILITY = 0.002
TEMPERATURE_RELAXATION = 0.12
HEAT_DIFFUSION = 0.15
CLIMATE_ITERATIONS_PER_STEP = 4
LOCAL_HEAT_FACTOR = 30.0
TEMPERATURE_NOISE = 0.05
CLIMATE_SPINUP_STEPS = 100
PROGRESS_INTERVAL = 100

# Luminosity scan used to locate the black-to-white dominance
# transition. Increase SCAN_POINTS and the step counts for a
# finer, less noisy estimate.
RUN_CRITICAL_SCAN = True
SCAN_POINTS = 41
SCAN_EQUILIBRATION_STEPS = 25
SCAN_SAMPLE_STEPS = 15
SCAN_MIN_DAISY_COVERAGE = 0.02
SCAN_SMOOTHING_WINDOW = 5

RANDOM_SEED = 42
STABILITY_TOLERANCE = 1.0e-12

# ---------------------- PHYSICAL CONSTANTS --------------------
STEFAN_BOLTZMANN = 5.670374419e-8
SOLAR_FLUX = 917.0

rng = np.random.default_rng(RANDOM_SEED)


def climate_stability_numbers():
    positivity_number = 4.0 * HEAT_DIFFUSION + TEMPERATURE_RELAXATION
    spectral_number = 8.0 * HEAT_DIFFUSION + TEMPERATURE_RELAXATION
    return positivity_number, spectral_number


def validate_parameters():
    if GRID_SIZE < 5:
        raise ValueError("GRID_SIZE must be at least 5.")
    if STEPS < 2:
        raise ValueError("STEPS must be at least 2.")
    if CLIMATE_ITERATIONS_PER_STEP < 1:
        raise ValueError("CLIMATE_ITERATIONS_PER_STEP must be at least 1.")
    if CLIMATE_SPINUP_STEPS < 0:
        raise ValueError("CLIMATE_SPINUP_STEPS cannot be negative.")
    if PROGRESS_INTERVAL < 1:
        raise ValueError("PROGRESS_INTERVAL must be at least 1.")
    if LUMINOSITY_START <= 0.0 or LUMINOSITY_END <= 0.0:
        raise ValueError("Both luminosity limits must be greater than zero.")
    if not 0.0 <= INITIAL_BLACK_FRACTION <= 1.0:
        raise ValueError("INITIAL_BLACK_FRACTION must be between 0 and 1.")
    if not 0.0 <= INITIAL_WHITE_FRACTION <= 1.0:
        raise ValueError("INITIAL_WHITE_FRACTION must be between 0 and 1.")
    if INITIAL_BLACK_FRACTION + INITIAL_WHITE_FRACTION > 1.0:
        raise ValueError("Initial daisy fractions cannot add up to more than 1.")
    if GROWTH_COEFFICIENT <= 0.0:
        raise ValueError("GROWTH_COEFFICIENT must be greater than zero.")
    if not 0.0 <= DEATH_PROBABILITY <= 1.0:
        raise ValueError("DEATH_PROBABILITY must be between 0 and 1.")
    if not 0.0 <= MAX_TEMPERATURE_MORTALITY <= 1.0:
        raise ValueError("MAX_TEMPERATURE_MORTALITY must be between 0 and 1.")
    if DEATH_PROBABILITY + MAX_TEMPERATURE_MORTALITY > 1.0:
        raise ValueError("Total maximum mortality cannot exceed 1.")
    if not 0.0 <= COLONIZATION_PROBABILITY <= 1.0:
        raise ValueError("COLONIZATION_PROBABILITY must be between 0 and 1.")
    if not 0.0 <= LONG_DISTANCE_SEEDING_PROBABILITY <= 1.0:
        raise ValueError("LONG_DISTANCE_SEEDING_PROBABILITY must be between 0 and 1.")
    if not 0.0 < TEMPERATURE_RELAXATION <= 1.0:
        raise ValueError("TEMPERATURE_RELAXATION must be greater than 0 and no greater than 1.")
    if HEAT_DIFFUSION < 0.0:
        raise ValueError("HEAT_DIFFUSION cannot be negative.")
    if LOCAL_HEAT_FACTOR < 0.0:
        raise ValueError("LOCAL_HEAT_FACTOR cannot be negative.")
    if TEMPERATURE_NOISE < 0.0:
        raise ValueError("TEMPERATURE_NOISE cannot be negative.")
    for albedo in (ALBEDO_BLACK, ALBEDO_GROUND, ALBEDO_WHITE):
        if not 0.0 <= albedo < 1.0:
            raise ValueError("Every albedo must be at least 0 and less than 1.")
    if not ALBEDO_BLACK < ALBEDO_GROUND < ALBEDO_WHITE:
        raise ValueError("Expected ALBEDO_BLACK < ALBEDO_GROUND < ALBEDO_WHITE.")
    _, spectral = climate_stability_numbers()
    if spectral > 2.0 + STABILITY_TOLERANCE:
        raise ValueError(
            "Explicit climate update is linearly unstable: "
            "8 * HEAT_DIFFUSION + TEMPERATURE_RELAXATION must not exceed 2."
        )


def luminosity_at_step(step):
    fraction = step / max(STEPS - 1, 1)
    return float(LUMINOSITY_START + fraction * (LUMINOSITY_END - LUMINOSITY_START))


def initialize_world():
    world = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    values = rng.random(world.shape)
    world[values < INITIAL_BLACK_FRACTION] = 1
    world[(values >= INITIAL_BLACK_FRACTION) &
          (values < INITIAL_BLACK_FRACTION + INITIAL_WHITE_FRACTION)] = 2
    return world


def make_albedo_map(world):
    world_array = np.asarray(world)
    if world_array.ndim != 2:
        raise ValueError("The biological world must be a two-dimensional array.")
    if world_array.size == 0:
        raise ValueError("The biological world cannot be empty.")
    if np.issubdtype(world_array.dtype, np.bool_):
        raise ValueError("The biological lattice cannot use Boolean values.")
    if not np.all(np.isin(world_array, [0, 1, 2])):
        raise ValueError("The biological lattice may contain only 0, 1, and 2.")
    albedo = np.full(world_array.shape, ALBEDO_GROUND, dtype=float)
    albedo[world_array == 1] = ALBEDO_BLACK
    albedo[world_array == 2] = ALBEDO_WHITE
    return albedo


def radiative_equilibrium_temperature(luminosity, planetary_albedo):
    absorbed = SOLAR_FLUX * float(luminosity) * (1.0 - float(planetary_albedo))
    absorbed = max(absorbed, 1.0e-12)
    result = (absorbed / STEFAN_BOLTZMANN) ** 0.25 - 273.15
    if not np.isfinite(result):
        raise RuntimeError("Radiative equilibrium produced a non-finite temperature.")
    return float(result)


def calculate_local_target_temperature(luminosity, albedo_map):
    albedo_array = np.asarray(albedo_map, dtype=float)
    if albedo_array.ndim != 2 or albedo_array.size == 0:
        raise ValueError("Albedo map must be a nonempty 2D array.")
    if not np.all(np.isfinite(albedo_array)):
        raise RuntimeError("Albedo map contains a non-finite value.")
    mean_albedo = float(np.mean(albedo_array))
    global_temperature = radiative_equilibrium_temperature(luminosity, mean_albedo)
    target = global_temperature + LOCAL_HEAT_FACTOR * (mean_albedo - albedo_array)
    if not np.all(np.isfinite(target)):
        raise RuntimeError("Local target temperature contains a non-finite value.")
    return target, global_temperature, mean_albedo


def calculate_laplacian(field):
    return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 4.0 * field)


def update_temperature(temperature, target_temperature, include_noise=True):
    temperature_array = np.asarray(temperature, dtype=float)
    target_array = np.asarray(target_temperature, dtype=float)
    if temperature_array.ndim != 2:
        raise ValueError("The temperature field must be a 2D array.")
    if target_array.ndim != 2:
        raise ValueError("The target-temperature field must be a 2D array.")
    if temperature_array.size == 0:
        raise ValueError("The temperature field cannot be empty.")
    if target_array.shape != temperature_array.shape:
        raise ValueError("Temperature and target-temperature arrays must have the same shape.")
    if not np.all(np.isfinite(temperature_array)):
        raise RuntimeError("Climate update received a non-finite temperature.")
    if not np.all(np.isfinite(target_array)):
        raise RuntimeError("Climate update received a non-finite target temperature.")

    updated = temperature_array.copy()
    for _ in range(CLIMATE_ITERATIONS_PER_STEP):
        updated += (TEMPERATURE_RELAXATION * (target_array - updated) +
                    HEAT_DIFFUSION * calculate_laplacian(updated))
        if not np.all(np.isfinite(updated)):
            raise RuntimeError("Temperature field became unstable during climate update.")

    if include_noise and TEMPERATURE_NOISE > 0.0:
        noise = rng.normal(0.0, TEMPERATURE_NOISE, updated.shape)
        noise -= np.mean(noise)
        updated += noise

    if not np.all(np.isfinite(updated)):
        raise RuntimeError("Temperature field contains a non-finite value.")
    return updated


def update_lifeless_temperature(current_temperature, luminosity):
    target = radiative_equilibrium_temperature(luminosity, ALBEDO_GROUND)
    updated = float(current_temperature)
    for _ in range(CLIMATE_ITERATIONS_PER_STEP):
        updated += TEMPERATURE_RELAXATION * (target - updated)
    if not np.isfinite(updated):
        raise RuntimeError("Lifeless control temperature became non-finite.")
    return float(updated)


def growth_rate(temperature):
    temperature_array = np.asarray(temperature, dtype=float)
    if not np.all(np.isfinite(temperature_array)):
        raise RuntimeError("Growth-rate calculation received a non-finite temperature.")
    difference = OPTIMUM_TEMPERATURE - temperature_array
    if np.any(np.abs(difference) > 1.0e6):
        raise RuntimeError("Growth-rate calculation received an unrealistically large temperature difference.")
    growth = 1.0 - GROWTH_COEFFICIENT * difference ** 2
    if not np.all(np.isfinite(growth)):
        raise RuntimeError("Growth-rate calculation produced a non-finite value.")
    return np.clip(growth, 0.0, 1.0)


def mortality_rate(temperature):
    suitability = growth_rate(temperature)
    mortality = DEATH_PROBABILITY + MAX_TEMPERATURE_MORTALITY * (1.0 - suitability)
    if not np.all(np.isfinite(mortality)):
        raise RuntimeError("Mortality calculation produced a non-finite value.")
    return np.clip(mortality, 0.0, 1.0)


def neighbor_fraction(mask):
    mask = np.asarray(mask, dtype=bool)
    neighbors = sum(
        np.roll(np.roll(mask, row_shift, axis=0), col_shift, axis=1).astype(float)
        for row_shift, col_shift in (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1)
        )
    )
    return np.clip(neighbors / 8.0, 0.0, 1.0)


def reproduce_daisies(world, temperature):
    updated = np.asarray(world, dtype=np.int8).copy()
    temp = np.asarray(temperature, dtype=float)
    if updated.shape != temp.shape:
        raise ValueError("World and temperature arrays must have the same shape.")
    make_albedo_map(updated)  # validates world
    if not np.all(np.isfinite(temp)):
        raise RuntimeError("Biological update received a non-finite temperature.")

    black = updated == 1
    white = updated == 2
    mortality = mortality_rate(temp)
    updated[black & (rng.random(updated.shape) < mortality)] = 0
    updated[white & (rng.random(updated.shape) < mortality)] = 0

    black_survivors = updated == 1
    white_survivors = updated == 2
    empty = updated == 0
    suitability = growth_rate(temp)

    black_probability = COLONIZATION_PROBABILITY * neighbor_fraction(black_survivors) * suitability
    white_probability = COLONIZATION_PROBABILITY * neighbor_fraction(white_survivors) * suitability
    if LONG_DISTANCE_SEEDING_PROBABILITY > 0.0:
        black_probability += LONG_DISTANCE_SEEDING_PROBABILITY * suitability
        white_probability += LONG_DISTANCE_SEEDING_PROBABILITY * suitability

    black_probability[~empty] = 0.0
    white_probability[~empty] = 0.0
    black_probability = np.clip(black_probability, 0.0, 1.0)
    white_probability = np.clip(white_probability, 0.0, 1.0)

    black_success = (rng.random(updated.shape) < black_probability) & empty
    white_success = (rng.random(updated.shape) < white_probability) & empty
    updated[black_success & ~white_success] = 1
    updated[white_success & ~black_success] = 2

    competition = black_success & white_success
    if np.any(competition):
        black_strength = black_probability[competition]
        white_strength = white_probability[competition]
        total = black_strength + white_strength
        probability_black = np.full(black_strength.shape, 0.5, dtype=float)
        np.divide(black_strength, total, out=probability_black, where=total > 0.0)
        probability_black = np.clip(probability_black, 0.0, 1.0)
        black_wins = rng.random(probability_black.size) < probability_black
        rows, cols = np.where(competition)
        updated[rows[black_wins], cols[black_wins]] = 1
        updated[rows[~black_wins], cols[~black_wins]] = 2

    make_albedo_map(updated)  # final validation
    return updated


def run_simulation():
    validate_parameters()
    positivity_number, spectral_number = climate_stability_numbers()

    print("=" * 76)
    print("2D DAISYWORLD WITH HEAT DIFFUSION")
    print("=" * 76)
    print(f"Grid size                    : {GRID_SIZE} x {GRID_SIZE}")
    print(f"Simulation steps             : {STEPS}")
    print(f"Solar luminosity             : {LUMINOSITY_START:.2f} to {LUMINOSITY_END:.2f}")
    print(f"Climate positivity number    : {positivity_number:.3f}")
    print(f"Climate spectral number      : {spectral_number:.3f}")
    print(f"Minimum mortality            : {DEATH_PROBABILITY:.3f}")
    print(f"Maximum mortality            : {DEATH_PROBABILITY + MAX_TEMPERATURE_MORTALITY:.3f}")
    print(f"Random seed                  : {RANDOM_SEED}")
    print("=" * 76)
    if positivity_number > 1.0 + STABILITY_TOLERANCE:
        print("WARNING: The climate update is linearly stable but may show local oscillations.")

    world = initialize_world()
    initial_albedo = make_albedo_map(world)
    initial_temperature = radiative_equilibrium_temperature(
        LUMINOSITY_START, float(np.mean(initial_albedo))
    )
    temperature = np.full(world.shape, initial_temperature, dtype=float)
    lifeless_temperature = radiative_equilibrium_temperature(LUMINOSITY_START, ALBEDO_GROUND)

    print("\nClimate spin-up started...")
    for _ in range(CLIMATE_SPINUP_STEPS):
        target, _, _ = calculate_local_target_temperature(LUMINOSITY_START, make_albedo_map(world))
        temperature = update_temperature(temperature, target, include_noise=False)
        lifeless_temperature = update_lifeless_temperature(lifeless_temperature, LUMINOSITY_START)
    print("Climate spin-up completed.\nMain simulation started...\n")

    time_history = np.arange(STEPS)
    luminosity_history = np.zeros(STEPS)
    black_history = np.zeros(STEPS)
    white_history = np.zeros(STEPS)
    bare_history = np.zeros(STEPS)
    mean_temperature_history = np.zeros(STEPS)
    minimum_temperature_history = np.zeros(STEPS)
    maximum_temperature_history = np.zeros(STEPS)
    global_target_history = np.zeros(STEPS)
    planetary_albedo_history = np.zeros(STEPS)
    lifeless_temperature_history = np.zeros(STEPS)
    mean_growth_history = np.zeros(STEPS)
    mean_mortality_history = np.zeros(STEPS)
    daisy_mortality_history = np.full(STEPS, np.nan)

    snapshot_steps = {0, STEPS // 2, STEPS - 1}
    snapshots = {}

    for step in range(STEPS):
        luminosity = luminosity_at_step(step)
        target, global_target, mean_albedo = calculate_local_target_temperature(
            luminosity, make_albedo_map(world)
        )
        temperature = update_temperature(temperature, target, include_noise=True)
        lifeless_temperature = update_lifeless_temperature(lifeless_temperature, luminosity)

        black_mask = world == 1
        white_mask = world == 2
        bare_mask = world == 0
        daisy_mask = black_mask | white_mask
        local_growth = growth_rate(temperature)
        local_mortality = mortality_rate(temperature)

        black_fraction = float(np.mean(black_mask))
        white_fraction = float(np.mean(white_mask))
        bare_fraction = float(np.mean(bare_mask))
        mean_temperature = float(np.mean(temperature))
        daisy_mortality = (float(np.mean(local_mortality[daisy_mask]))
                           if np.any(daisy_mask) else np.nan)

        luminosity_history[step] = luminosity
        black_history[step] = black_fraction
        white_history[step] = white_fraction
        bare_history[step] = bare_fraction
        mean_temperature_history[step] = mean_temperature
        minimum_temperature_history[step] = float(np.min(temperature))
        maximum_temperature_history[step] = float(np.max(temperature))
        global_target_history[step] = global_target
        planetary_albedo_history[step] = mean_albedo
        lifeless_temperature_history[step] = lifeless_temperature
        mean_growth_history[step] = float(np.mean(local_growth))
        mean_mortality_history[step] = float(np.mean(local_mortality))
        daisy_mortality_history[step] = daisy_mortality

        if step in snapshot_steps:
            snapshots[step] = (world.copy(), temperature.copy(), luminosity)

        if step % PROGRESS_INTERVAL == 0 or step == STEPS - 1:
            mortality_text = f"{daisy_mortality:6.3f}" if np.isfinite(daisy_mortality) else "   N/A"
            print(
                f"Step {step + 1:4d}/{STEPS} | L = {luminosity:.3f} | "
                f"T = {mean_temperature:7.2f} C | Mortality = {mortality_text} | "
                f"Black = {black_fraction:6.3f} | White = {white_fraction:6.3f} | "
                f"Total = {black_fraction + white_fraction:6.3f}"
            )

        if step < STEPS - 1:
            world = reproduce_daisies(world, temperature)

    coverage_sum = black_history + white_history + bare_history
    if not np.allclose(coverage_sum, 1.0, atol=1.0e-10):
        raise RuntimeError("Recorded surface fractions do not add up to 1.")

    finite_histories = (
        luminosity_history, black_history, white_history, bare_history,
        mean_temperature_history, minimum_temperature_history,
        maximum_temperature_history, global_target_history,
        planetary_albedo_history, lifeless_temperature_history,
        mean_growth_history, mean_mortality_history
    )
    if any(not np.all(np.isfinite(history)) for history in finite_histories):
        raise RuntimeError("A simulation history contains a non-finite value.")
    if np.any(np.isinf(daisy_mortality_history)):
        raise RuntimeError("Occupied-cell mortality contains infinity.")

    finite_mortality = daisy_mortality_history[np.isfinite(daisy_mortality_history)]
    if np.any((finite_mortality < 0.0) | (finite_mortality > 1.0)):
        raise RuntimeError("Occupied-cell mortality is outside [0, 1].")

    total_daisy_history = black_history + white_history
    extinct = np.isclose(total_daisy_history, 0.0, atol=1.0e-12, rtol=0.0)
    undefined = np.isnan(daisy_mortality_history)
    if np.any(undefined & ~extinct) or np.any(extinct & ~undefined):
        raise RuntimeError("Extinction and occupied-cell mortality histories are inconsistent.")

    maximum_index = int(np.argmax(total_daisy_history))
    living_minus_lifeless = mean_temperature_history - lifeless_temperature_history
    difference_index = int(np.argmax(np.abs(living_minus_lifeless)))
    max_mortality = float(np.max(finite_mortality)) if finite_mortality.size else np.nan

    print("\nSimulation completed.")
    print("=" * 76)
    print("FINAL RESULTS")
    print("=" * 76)
    print(f"Final luminosity              : {luminosity_history[-1]:.3f}")
    print(f"Final mean temperature        : {mean_temperature_history[-1]:.2f} C")
    print(f"Final lifeless temperature    : {lifeless_temperature_history[-1]:.2f} C")
    print(f"Final black coverage          : {black_history[-1]:.3f}")
    print(f"Final white coverage          : {white_history[-1]:.3f}")
    print(f"Final bare coverage           : {bare_history[-1]:.3f}")
    print(f"Maximum daisy coverage        : {total_daisy_history[maximum_index]:.3f}")
    print(f"Luminosity at maximum         : {luminosity_history[maximum_index]:.3f}")
    print(f"Largest living-control diff   : {living_minus_lifeless[difference_index]:+.2f} C")
    print(f"Luminosity at largest diff    : {luminosity_history[difference_index]:.3f}")
    print(f"Maximum recorded mortality    : {max_mortality:.3f}" if np.isfinite(max_mortality)
          else "Maximum recorded mortality    : N/A")
    print("=" * 76)

    histories = {
        "time": time_history,
        "luminosity": luminosity_history,
        "black": black_history,
        "white": white_history,
        "bare": bare_history,
        "mean_temperature": mean_temperature_history,
        "min_temperature": minimum_temperature_history,
        "max_temperature": maximum_temperature_history,
        "global_target": global_target_history,
        "albedo": planetary_albedo_history,
        "lifeless_temperature": lifeless_temperature_history,
        "mean_growth": mean_growth_history,
        "mean_mortality": mean_mortality_history,
        "daisy_mortality": daisy_mortality_history,
    }
    return histories, snapshots


def plot_results(histories, snapshots):
    luminosity = histories["luminosity"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    axes[0, 0].plot(histories["time"], luminosity, color="darkorange", linewidth=2.5)
    axes[0, 0].set(title="Slowly Changing Solar Luminosity", xlabel="Simulation step", ylabel="Relative luminosity")

    axes[0, 1].plot(luminosity, histories["black"], color="black", linewidth=2, label="Black daisies")
    axes[0, 1].plot(luminosity, histories["white"], color="royalblue", linewidth=2, label="White daisies")
    axes[0, 1].plot(luminosity, histories["bare"], color="saddlebrown", linestyle="--", label="Bare ground")
    axes[0, 1].set(title="Surface Coverage vs Solar Luminosity", xlabel="Solar luminosity", ylabel="Coverage fraction", ylim=(0, 1))
    axes[0, 1].legend()

    axes[1, 0].plot(luminosity, histories["mean_temperature"], color="red", linewidth=2.5, label="Living Daisyworld")
    axes[1, 0].plot(luminosity, histories["lifeless_temperature"], color="gray", linestyle="--", label="Lifeless control")
    axes[1, 0].plot(luminosity, histories["global_target"], color="darkorange", linestyle="-.", label="Radiative target")
    axes[1, 0].axhline(OPTIMUM_TEMPERATURE, color="green", linestyle=":", label="Optimal temperature")
    axes[1, 0].fill_between(luminosity, histories["min_temperature"], histories["max_temperature"], color="red", alpha=0.12)
    axes[1, 0].set(title="Planetary Temperature Regulation", xlabel="Solar luminosity", ylabel="Temperature (C)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(luminosity, histories["albedo"], color="purple", linewidth=2.5, label="Living planetary albedo")
    axes[1, 1].axhline(ALBEDO_GROUND, color="saddlebrown", linestyle="--", label="Bare-ground albedo")
    axes[1, 1].set(title="Planetary Albedo Feedback", xlabel="Solar luminosity", ylabel="Mean planetary albedo")
    axes[1, 1].legend()

    axes[2, 0].plot(luminosity, histories["mean_growth"], color="green", linewidth=2.5)
    axes[2, 0].set(title="Mean Temperature Growth Suitability", xlabel="Solar luminosity", ylabel="Mean growth suitability", ylim=(0, 1.05))

    axes[2, 1].plot(luminosity, histories["mean_mortality"], color="orange", label="Whole-lattice mortality")
    axes[2, 1].plot(luminosity, histories["daisy_mortality"], color="darkred", linewidth=2.5, label="Mortality at occupied cells")
    axes[2, 1].axhline(DEATH_PROBABILITY, color="black", linestyle=":", label="Baseline mortality")
    axes[2, 1].set(title="Temperature-Dependent Mortality", xlabel="Solar luminosity", ylabel="Mortality probability", ylim=(0, min(1.0, DEATH_PROBABILITY + MAX_TEMPERATURE_MORTALITY + 0.05)))
    axes[2, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(True)
    fig.suptitle("2D Daisyworld: Climate, Ecology, and Mortality", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    steps = [0, STEPS // 2, STEPS - 1]
    colors = ["#9b7653", "#111111", "#f2f2f2"]
    cmap = ListedColormap(colors)
    fig_maps, map_axes = plt.subplots(2, 3, figsize=(15, 9))
    all_temperatures = [snapshots[step][1] for step in steps]
    t_min = min(float(np.min(field)) for field in all_temperatures)
    t_max = max(float(np.max(field)) for field in all_temperatures)
    if np.isclose(t_min, t_max):
        t_min -= 0.5
        t_max += 0.5

    labels = ["Initial", "Middle", "Final"]
    temperature_image = None
    for column, (step, label) in enumerate(zip(steps, labels)):
        world, temperature, luminosity_value = snapshots[step]
        map_axes[0, column].imshow(world, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        map_axes[0, column].set_title(f"{label} World\nL = {luminosity_value:.2f}")
        temperature_image = map_axes[1, column].imshow(temperature, cmap="coolwarm", vmin=t_min, vmax=t_max, interpolation="nearest")
        map_axes[1, column].set_title(f"{label} Temperature Field")
        for row in (0, 1):
            map_axes[row, column].set_xticks([])
            map_axes[row, column].set_yticks([])

    legend = [
        Patch(facecolor=colors[0], edgecolor="black", label="Bare ground"),
        Patch(facecolor=colors[1], edgecolor="black", label="Black daisies"),
        Patch(facecolor=colors[2], edgecolor="black", label="White daisies"),
    ]
    fig_maps.legend(handles=legend, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94))
    colorbar = fig_maps.colorbar(temperature_image, ax=map_axes[1, :], orientation="horizontal", fraction=0.06, pad=0.10)
    colorbar.set_label("Temperature (C)")
    fig_maps.suptitle("Spatial Evolution of 2D Daisyworld", fontsize=16)
    fig_maps.subplots_adjust(top=0.86, bottom=0.15, wspace=0.10, hspace=0.20)



def moving_average(values, window):
    """Return a centered moving average with edge padding."""
    values = np.asarray(values, dtype=float)
    window = int(window)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def scan_luminosity_for_critical_point():
    """
    Perform a quasi-static upward luminosity scan.

    At each luminosity, the model first equilibrates and is then
    sampled. The order parameter is

        M = black coverage - white coverage.

    Two transition estimates are reported:
      1. dominance crossover, where M changes sign;
      2. response point, where |dM/dL| is largest while daisy
         coverage remains above SCAN_MIN_DAISY_COVERAGE.

    These are operational finite-grid estimates, not an exact
    thermodynamic critical point.
    """
    global rng

    if SCAN_POINTS < 3:
        raise ValueError("SCAN_POINTS must be at least 3.")
    if SCAN_EQUILIBRATION_STEPS < 1 or SCAN_SAMPLE_STEPS < 1:
        raise ValueError("Critical-scan step counts must be positive.")
    if not 0.0 <= SCAN_MIN_DAISY_COVERAGE <= 1.0:
        raise ValueError("SCAN_MIN_DAISY_COVERAGE must be between 0 and 1.")

    # Use a distinct deterministic stream so the scan does not
    # depend on how many random numbers the main run consumed.
    rng = np.random.default_rng(RANDOM_SEED + 1000)
    luminosities = np.linspace(LUMINOSITY_START, LUMINOSITY_END, SCAN_POINTS)

    world = initialize_world()
    initial_albedo = make_albedo_map(world)
    initial_temperature = radiative_equilibrium_temperature(
        luminosities[0], float(np.mean(initial_albedo))
    )
    temperature = np.full(world.shape, initial_temperature, dtype=float)

    # Initial climate spin-up at the first luminosity.
    for _ in range(CLIMATE_SPINUP_STEPS):
        target, _, _ = calculate_local_target_temperature(
            luminosities[0], make_albedo_map(world)
        )
        temperature = update_temperature(temperature, target, include_noise=False)

    black_mean = np.zeros(SCAN_POINTS)
    white_mean = np.zeros(SCAN_POINTS)
    total_mean = np.zeros(SCAN_POINTS)
    magnetization_mean = np.zeros(SCAN_POINTS)
    temperature_mean = np.zeros(SCAN_POINTS)

    print("\nCritical-point luminosity scan started...")

    for index, luminosity in enumerate(luminosities):
        # Quasi-static continuation: the final state at one
        # luminosity becomes the initial state at the next.
        for _ in range(SCAN_EQUILIBRATION_STEPS):
            target, _, _ = calculate_local_target_temperature(
                luminosity, make_albedo_map(world)
            )
            temperature = update_temperature(temperature, target, include_noise=True)
            world = reproduce_daisies(world, temperature)

        black_samples = []
        white_samples = []
        temperature_samples = []

        for _ in range(SCAN_SAMPLE_STEPS):
            target, _, _ = calculate_local_target_temperature(
                luminosity, make_albedo_map(world)
            )
            temperature = update_temperature(temperature, target, include_noise=True)
            world = reproduce_daisies(world, temperature)
            black_samples.append(float(np.mean(world == 1)))
            white_samples.append(float(np.mean(world == 2)))
            temperature_samples.append(float(np.mean(temperature)))

        black_mean[index] = float(np.mean(black_samples))
        white_mean[index] = float(np.mean(white_samples))
        total_mean[index] = black_mean[index] + white_mean[index]
        magnetization_mean[index] = black_mean[index] - white_mean[index]
        temperature_mean[index] = float(np.mean(temperature_samples))

        print(
            f"Scan {index + 1:3d}/{SCAN_POINTS} | "
            f"L = {luminosity:.3f} | "
            f"M = {magnetization_mean[index]:+.4f} | "
            f"Daisies = {total_mean[index]:.4f}"
        )

    smoothed_magnetization = moving_average(
        magnetization_mean, SCAN_SMOOTHING_WINDOW
    )
    derivative = np.gradient(smoothed_magnetization, luminosities)
    viable = total_mean >= SCAN_MIN_DAISY_COVERAGE

    if np.any(viable):
        viable_indices = np.flatnonzero(viable)
        response_index = viable_indices[
            np.argmax(np.abs(derivative[viable_indices]))
        ]
        response_luminosity = float(luminosities[response_index])
    else:
        response_index = None
        response_luminosity = np.nan

    crossover_luminosity = np.nan
    crossover_index = None
    for index in range(SCAN_POINTS - 1):
        if not (viable[index] and viable[index + 1]):
            continue
        m1 = smoothed_magnetization[index]
        m2 = smoothed_magnetization[index + 1]
        if m1 == 0.0:
            crossover_luminosity = float(luminosities[index])
            crossover_index = index
            break
        if m1 * m2 < 0.0:
            l1 = luminosities[index]
            l2 = luminosities[index + 1]
            crossover_luminosity = float(
                l1 - m1 * (l2 - l1) / (m2 - m1)
            )
            crossover_index = index
            break

    print("\nCRITICAL-POINT SCAN RESULTS")
    print("=" * 76)
    if np.isfinite(crossover_luminosity):
        print(
            f"Black-white dominance crossover : "
            f"L = {crossover_luminosity:.4f}"
        )
    else:
        print("Black-white dominance crossover : not found in viable range")
    if np.isfinite(response_luminosity):
        print(
            f"Maximum-response estimate        : "
            f"L = {response_luminosity:.4f}"
        )
        print(
            f"Order parameter at response      : "
            f"M = {smoothed_magnetization[response_index]:+.4f}"
        )
        print(
            f"Daisy coverage at response       : "
            f"{total_mean[response_index]:.4f}"
        )
    else:
        print("Maximum-response estimate        : unavailable; no viable daisies")
    print("=" * 76)

    results = {
        "luminosity": luminosities,
        "black": black_mean,
        "white": white_mean,
        "total": total_mean,
        "magnetization": magnetization_mean,
        "smoothed_magnetization": smoothed_magnetization,
        "derivative": derivative,
        "temperature": temperature_mean,
        "viable": viable,
        "crossover_luminosity": crossover_luminosity,
        "response_luminosity": response_luminosity,
    }
    return results


def plot_critical_scan(scan):
    """Plot the luminosity scan and mark the transition safely."""
    required = {
        "luminosity", "black", "white", "total", "magnetization",
        "smoothed_magnetization", "derivative", "temperature",
        "crossover_luminosity", "response_luminosity"
    }
    missing = required.difference(scan)
    if missing:
        raise KeyError(f"Critical-scan results are missing: {sorted(missing)}")

    luminosity = np.asarray(scan["luminosity"], dtype=float)
    black = np.asarray(scan["black"], dtype=float)
    white = np.asarray(scan["white"], dtype=float)
    total = np.asarray(scan["total"], dtype=float)
    raw_m = np.asarray(scan["magnetization"], dtype=float)
    smooth_m = np.asarray(scan["smoothed_magnetization"], dtype=float)
    derivative = np.asarray(scan["derivative"], dtype=float)
    temperature = np.asarray(scan["temperature"], dtype=float)

    arrays = (black, white, total, raw_m, smooth_m, derivative, temperature)
    if luminosity.ndim != 1 or luminosity.size < 2:
        raise ValueError("Critical scan needs at least two luminosity points.")
    if any(array.shape != luminosity.shape for array in arrays):
        raise ValueError("All critical-scan arrays must have the same 1D shape.")
    if not np.all(np.isfinite(luminosity)):
        raise ValueError("Luminosity scan contains a non-finite value.")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("A critical-scan result contains a non-finite value.")
    if np.any(np.diff(luminosity) <= 0.0):
        raise ValueError("Luminosity scan values must be strictly increasing.")

    crossover = float(scan["crossover_luminosity"])
    response = float(scan["response_luminosity"])
    scan_resolution = float(np.median(np.diff(luminosity)))
    crossover_uncertainty = 0.5 * scan_resolution

    m_min = float(np.min(smooth_m))
    m_max = float(np.max(smooth_m))
    m_span = m_max - m_min
    if np.isclose(m_span, 0.0):
        m_padding = 0.1
    else:
        m_padding = 0.08 * m_span
    phase_ymin = min(m_min - m_padding, -0.05)
    phase_ymax = max(m_max + m_padding, 0.05)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(luminosity, black, color="black", linewidth=2, label="Black daisies")
    axes[0, 0].plot(luminosity, white, color="royalblue", linewidth=2, label="White daisies")
    axes[0, 0].plot(luminosity, total, color="green", linestyle="--", linewidth=2, label="Total coverage")
    axes[0, 0].set(title="Equilibrium Daisy Coverage", xlabel="Luminosity", ylabel="Coverage")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].legend()

    axes[0, 1].plot(luminosity, raw_m, color="silver", linewidth=1.2, label="Raw M")
    axes[0, 1].plot(luminosity, smooth_m, color="purple", linewidth=3, label="Smoothed M")
    axes[0, 1].axhline(0.0, color="black", linestyle=":", linewidth=1.5)
    axes[0, 1].set_ylim(phase_ymin, phase_ymax)

    if np.isfinite(crossover):
        axes[0, 1].axvspan(
            luminosity[0], crossover, color="black", alpha=0.07,
            label="Black-daisy phase"
        )
        axes[0, 1].axvspan(
            crossover, luminosity[-1], color="royalblue", alpha=0.07,
            label="White-daisy phase"
        )
        axes[0, 1].axvline(
            crossover, color="blue", linewidth=3, linestyle="--",
            label=f"Crossover Lc = {crossover:.3f}"
        )
        text_x = min(
            crossover + 0.06 * (luminosity[-1] - luminosity[0]),
            luminosity[-1] - 0.02 * (luminosity[-1] - luminosity[0])
        )
        text_y = phase_ymin + 0.78 * (phase_ymax - phase_ymin)
        axes[0, 1].annotate(
            f"Lc = {crossover:.3f} +/- {crossover_uncertainty:.3f}",
            xy=(crossover, 0.0), xytext=(text_x, text_y),
            arrowprops=dict(arrowstyle="->", color="blue", linewidth=1.5),
            color="blue", fontsize=10, fontweight="bold",
            horizontalalignment="right" if text_x >= crossover else "left"
        )

    if np.isfinite(response):
        axes[0, 1].axvline(
            response, color="red", linewidth=2, linestyle="-.",
            label=f"Maximum response = {response:.3f}"
        )

    axes[0, 1].set(
        title="Black-to-White Dominance Transition",
        xlabel="Luminosity", ylabel="M = Black - White"
    )
    axes[0, 1].legend(fontsize=8, loc="best")

    axes[1, 0].plot(luminosity, derivative, color="darkred", linewidth=2.5, label="dM/dL")
    if np.isfinite(crossover):
        axes[1, 0].axvline(crossover, color="blue", linestyle="--", linewidth=2, label=f"Crossover = {crossover:.3f}")
    if np.isfinite(response):
        axes[1, 0].axvline(response, color="red", linestyle="-.", linewidth=2, label=f"Maximum response = {response:.3f}")
    axes[1, 0].set(title="Response Function dM/dL", xlabel="Luminosity", ylabel="dM/dL")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(luminosity, temperature, color="red", linewidth=2.5, label="Mean temperature")
    axes[1, 1].axhline(OPTIMUM_TEMPERATURE, color="green", linestyle=":", label="Growth optimum")
    if np.isfinite(crossover):
        axes[1, 1].axvline(crossover, color="blue", linestyle="--", linewidth=2, label=f"Crossover = {crossover:.3f}")
    if np.isfinite(response):
        axes[1, 1].axvline(response, color="red", linestyle="-.", linewidth=2, label=f"Maximum response = {response:.3f}")
    axes[1, 1].set(title="Temperature During Luminosity Scan", xlabel="Luminosity", ylabel="Temperature (C)")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(True)

    fig.suptitle("Daisyworld Critical-Point Analysis", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    transition_fig, transition_ax = plt.subplots(figsize=(10, 6))
    transition_ax.plot(luminosity, raw_m, color="silver", linewidth=1.2, label="Raw M")
    transition_ax.plot(luminosity, smooth_m, color="purple", linewidth=3, label="Smoothed M")
    transition_ax.axhline(0.0, color="black", linestyle=":", linewidth=1.5)
    transition_ax.set_ylim(phase_ymin, phase_ymax)

    if np.isfinite(crossover):
        transition_ax.axvspan(luminosity[0], crossover, color="black", alpha=0.07, label="Black-daisy phase")
        transition_ax.axvspan(crossover, luminosity[-1], color="royalblue", alpha=0.07, label="White-daisy phase")
        transition_ax.axvline(crossover, color="blue", linestyle="--", linewidth=3, label=f"Lc = {crossover:.3f} +/- {crossover_uncertainty:.3f}")

    if np.isfinite(response):
        transition_ax.axvline(response, color="red", linestyle="-.", linewidth=2, label=f"Maximum response = {response:.3f}")

    transition_ax.set_xlabel("Luminosity")
    transition_ax.set_ylabel("M = Black - White")
    transition_ax.set_title("Black-to-White Daisy Dominance Transition")
    transition_ax.grid(True)
    transition_ax.legend(fontsize=9, loc="best")
    transition_fig.tight_layout()

def main():
    histories, snapshots = run_simulation()
    plot_results(histories, snapshots)

    if RUN_CRITICAL_SCAN:
        scan_results = scan_luminosity_for_critical_point()
        plot_critical_scan(scan_results)

    print("\nOpening result graphs...")
    print("Close both graph windows to return to the console.")
    plt.show()
    input("\nPress Enter to exit the program...")


if __name__ == "__main__":
    main()