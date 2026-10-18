"""Read and check the action, parameter, and reward settings of a bandit environment."""

import copy

import numpy as np

from .reward_models.noise import noise_moment_root


def fields(mapping, allowed, label):
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"{label}: unknown fields {sorted(unknown)}")


def integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def positive(value, label, *, zero=False):
    result = number(value, label)
    if result < 0 if zero else result <= 0:
        raise ValueError(f"{label} has an invalid value")
    return result


def choice(value, allowed, label):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{label} must be one of {sorted(allowed)}")
    return value


def resolve_noise(raw, feedback_kind):
    noise = copy.deepcopy(raw)
    fields(
        noise,
        {
            "distribution",
            "scale",
            "degrees_of_freedom",
            "shape",
            "moment_exponent",
            "known_moment",
            "scale_mode",
            "max_log10_scale",
            "moment_root_bound",
            "centered",
        },
        "noise",
    )
    choice(feedback_kind, {"additive", "bernoulli", "categorical"}, "feedback kind")
    categorical = feedback_kind in {"bernoulli", "categorical"}
    distribution = noise.setdefault("distribution", "none" if categorical else "gaussian")
    choice(
        distribution,
        {"none", "gaussian", "student-t", "pareto", "lomax", "fisk"},
        "noise distribution",
    )
    if categorical and distribution != "none":
        raise ValueError("categorical logistic feedback cannot have additive real-valued noise")
    noise.setdefault("centered", True)
    if not isinstance(noise["centered"], bool):
        raise ValueError("noise centered must be true or false")
    if not noise["centered"] and distribution not in {"pareto", "lomax", "fisk"}:
        raise ValueError("centered=false applies only to Pareto, Lomax or Fisk noise")
    noise["scale"] = positive(
        noise.get("scale", 0.0 if distribution == "none" else 0.1), "noise scale", zero=True
    )
    if distribution == "none" and noise["scale"] != 0:
        raise ValueError("zero noise requires scale=0")
    noise["moment_exponent"] = positive(noise.get("moment_exponent", 0.5), "moment_exponent")
    if noise["moment_exponent"] > 1:
        raise ValueError("moment_exponent must be in (0, 1]")
    noise.setdefault("known_moment", False)
    if not isinstance(noise["known_moment"], bool):
        raise ValueError("known_moment must be true or false")
    if categorical and noise["known_moment"]:
        raise ValueError("additive noise moment metadata does not apply to categorical feedback")
    noise.setdefault("scale_mode", "constant")
    choice(noise["scale_mode"], {"constant", "log-uniform"}, "noise scale_mode")
    noise["max_log10_scale"] = positive(
        noise.get("max_log10_scale", 0.0), "max_log10_scale", zero=True
    )
    if noise["scale_mode"] == "constant" and noise["max_log10_scale"] != 0:
        raise ValueError("constant scale requires max_log10_scale=0")
    if categorical and noise["scale_mode"] != "constant":
        raise ValueError("categorical feedback does not support additive scale variation")
    noise.setdefault("degrees_of_freedom", 3.0 if distribution == "student-t" else None)
    noise.setdefault("shape", 3.0 if distribution in {"pareto", "lomax", "fisk"} else None)
    if distribution != "student-t" and noise["degrees_of_freedom"] is not None:
        raise ValueError("degrees_of_freedom applies only to student-t noise")
    if distribution not in {"pareto", "lomax", "fisk"} and noise["shape"] is not None:
        raise ValueError("shape applies only to pareto, lomax or fisk noise")
    if distribution == "student-t":
        noise["degrees_of_freedom"] = positive(noise["degrees_of_freedom"], "degrees_of_freedom")
    if distribution in {"pareto", "lomax", "fisk"}:
        noise["shape"] = positive(noise["shape"], "noise shape")
    root = noise_moment_root(
        distribution, noise["moment_exponent"], df=noise["degrees_of_freedom"],
        shape=noise["shape"], centered=noise["centered"]
    )
    try:
        bound = noise["scale"] * 10.0 ** noise["max_log10_scale"] * root
    except OverflowError as exc:
        raise ValueError("configured noise scale is not representable") from exc
    if not np.isfinite(bound):
        raise ValueError("configured noise moment bound is not finite")
    if "moment_root_bound" in noise:
        supplied_bound = number(noise["moment_root_bound"], "moment_root_bound")
        if not np.isclose(supplied_bound, bound, rtol=1e-12, atol=1e-14):
            raise ValueError("moment_root_bound is derived from noise parameters, not user-tuned")
    noise["moment_root_bound"] = float(bound)
    return noise


def resolve_environment(raw):
    """Check the environment settings and fill in omitted defaults."""
    env = copy.deepcopy(raw)
    fields(
        env,
        {"dimension", "mean_model", "feedback", "noise", "parameter_process", "action_process"},
        "environment",
    )
    d = env["dimension"] = integer(env.get("dimension", 2), "dimension")
    mean = env.setdefault("mean_model", {})
    fields(mean, {"link", "outcomes"}, "mean_model")
    link = mean.setdefault("link", "identity")
    choice(link, {"identity", "sigmoid", "softmax"}, "mean_model link")
    if link == "softmax":
        k = mean["outcomes"] = integer(mean.get("outcomes", 3), "mean_model outcomes")
    else:
        if "outcomes" in mean:
            raise ValueError("mean_model outcomes is category-only and requires softmax")
        k = 1
    feedback = env.setdefault("feedback", {})
    fields(feedback, {"kind", "reward_vector"}, "feedback")
    kind = feedback.setdefault(
        "kind", {"identity": "additive", "sigmoid": "bernoulli", "softmax": "categorical"}[link]
    )
    choice(kind, {"additive", "bernoulli", "categorical"}, "feedback kind")
    valid_pairs = {
        ("identity", "additive"),
        ("sigmoid", "additive"),
        ("sigmoid", "bernoulli"),
        ("softmax", "categorical"),
    }
    if (link, kind) not in valid_pairs:
        raise ValueError(f"unsupported mean_model link {link!r} with feedback {kind!r}")
    if kind == "additive":
        if "reward_vector" in feedback:
            raise ValueError(
                "feedback reward_vector is category-only; additive rewards are real-valued"
            )
    else:
        feedback.setdefault("reward_vector", (np.arange(k + 1) / k).tolist())
        try:
            rho = np.asarray(feedback["reward_vector"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("feedback reward_vector must be numeric") from exc
        if rho.shape != (k + 1,) or not np.isfinite(rho).all():
            raise ValueError("feedback reward_vector must contain outcomes+1 finite values")
        if kind == "bernoulli" and not np.array_equal(rho, [0.0, 1.0]):
            raise ValueError("bernoulli feedback requires canonical reward_vector=[0,1]")
        if rho[0] != 0 or np.any(rho < 0) or not np.any(rho[1:] > 0):
            raise ValueError(
                "reward_vector must have baseline 0 and nonnegative rewards, some positive"
            )
        feedback["reward_vector"] = rho.tolist()
    parameter = env.setdefault("parameter_process", {})
    fields(parameter, {"kind", "bound", "initial", "turns"}, "parameter_process")
    process_kind = parameter.setdefault("kind", "stationary")
    choice(process_kind, {"stationary", "rotation"}, "parameter_process kind")
    s = parameter["bound"] = positive(
        parameter.get("bound", 1.0), "parameter_process bound", zero=True
    )
    parameter["turns"] = number(
        parameter.get("turns", 0.0 if process_kind == "stationary" else 1.0),
        "parameter_process turns",
    )
    if process_kind == "stationary" and parameter["turns"] != 0:
        raise ValueError("stationary parameter_process requires turns=0")
    if process_kind == "rotation" and d < 2:
        raise ValueError("rotation requires dimension>=2")
    if link == "softmax":
        if "initial" not in parameter:
            theta = np.zeros((k, d))
            theta[:, 0] = np.cos(2 * np.pi * np.arange(k) / k)
            if d > 1:
                theta[:, 1] = np.sin(2 * np.pi * np.arange(k) / k)
            theta *= s / np.linalg.norm(theta) if s else 0.0
            parameter["initial"] = theta.tolist()
        expected_shape = (k, d)
    else:
        parameter.setdefault("initial", [s] + [0.0] * (d - 1))
        expected_shape = (d,)
    try:
        theta = np.asarray(parameter["initial"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("parameter_process initial must be a numeric vector or matrix") from exc
    if theta.shape != expected_shape or not np.isfinite(theta).all():
        raise ValueError(f"parameter_process initial must have finite shape {expected_shape}")
    if np.linalg.norm(theta) > s + 1e-12:
        raise ValueError("parameter Euclidean/Frobenius norm exceeds parameter_process bound")
    parameter["initial"] = theta.tolist()
    actions = env.setdefault("action_process", {})
    fields(
        actions,
        {"mode", "sampler", "count", "bound", "pool_size", "resample_per_trial", "seed"},
        "action_process",
    )
    n = actions["count"] = integer(actions.get("count", 8), "action_process count")
    actions["bound"] = positive(actions.get("bound", 1.0), "action_process bound")
    sampler = actions.setdefault("sampler", "gaussian-sphere")
    choice(
        sampler,
        {"gaussian-sphere", "uniform-cube-normalized", "uniform-cube-clipped"},
        "action_process sampler",
    )
    mode = actions.setdefault("mode", "fixed")
    choice(mode, {"fixed", "resample", "subset"}, "action_process mode")
    actions["pool_size"] = integer(actions.get("pool_size", n), "pool_size")
    if actions["pool_size"] < n:
        raise ValueError("pool_size must be >= action_process count")
    if mode != "subset" and actions["pool_size"] != n:
        raise ValueError("pool_size differs from count only for subset action sets")
    actions.setdefault("resample_per_trial", True)
    if not isinstance(actions["resample_per_trial"], bool):
        raise ValueError("resample_per_trial must be true or false")
    actions["seed"] = integer(actions.get("seed", 0), "action_process seed", 0)
    if actions["resample_per_trial"] and actions["seed"] != 0:
        raise ValueError("action_process seed applies only when resample_per_trial=false")
    env["noise"] = resolve_noise(env.get("noise", {}), kind)
    return env
