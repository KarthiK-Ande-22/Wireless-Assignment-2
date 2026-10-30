"""Numbers read from the SER curves: SNR at a target SER, slopes, deviations."""
import numpy as np

from .plotting import MIN_ERRORS


def snr_at_ser(snr_db, ser, n_errors, target):
    """SNR (dB) where a simulated curve reaches `target` (NaN if it does not)."""
    reliable = n_errors >= MIN_ERRORS
    log_ser, snr = np.log10(ser[reliable]), snr_db[reliable]
    if not log_ser.min() <= np.log10(target) <= log_ser.max():
        return np.nan
    return np.interp(-np.log10(target), -log_ser, snr)   # linear interpolation in log10(SER)


def theory_snr_at_ser(ser_function, target, snr_db_max=80.0):
    """SNR (dB) where a theoretical curve reaches `target`."""
    snr_db = np.arange(-10.0, snr_db_max, 0.01)
    ser = ser_function(10 ** (snr_db / 10))
    ok = ser > 0   # the AWGN SER becomes exactly 0 at very high SNR
    return np.interp(-np.log10(target), -np.log10(ser[ok]), snr_db[ok])


def high_snr_slopes(snr_db, ser, n_errors, n_segments=3, min_errors=50):
    """Slopes -dlog10(SER)/dlog10(SNR) of the last steps of a curve, and their SNR points."""
    reliable = n_errors >= min_errors
    snr = snr_db[reliable][-(n_segments + 1):]
    log_ser = np.log10(ser[reliable][-(n_segments + 1):])
    return -np.diff(log_ser) / np.diff(snr / 10), snr


def loglog_slope(snr_db, ser):
    """Least-squares slope -dlog10(SER)/dlog10(SNR) through the given points."""
    return -np.polyfit(np.asarray(snr_db) / 10, np.log10(ser), 1)[0]


def fitted_slope(snr_db, ser, n_errors, snr_min_db):
    """Least-squares slope over the plotted points with SNR >= snr_min_db."""
    ok = (n_errors >= MIN_ERRORS) & (snr_db >= snr_min_db)
    return loglog_slope(snr_db[ok], ser[ok])


def z_score(p_a, p_b, variance):
    """Difference of two SER values, in standard deviations."""
    return (p_a - p_b) / np.sqrt(variance)


def agreement_lines(comparisons, n_symbols):
    """Text table: how closely each simulated curve follows its theory."""
    lines = [f"Agreement with theory over the plotted points (>= {MIN_ERRORS} errors), "
             "sigma = sqrt(P (1 - P) / N)",
             f"{'simulation vs theory':>36}{'points':>8}{'mean dev.':>11}{'max |dev.|':>12}{'max |z|':>9}"]
    for label, ser, n_errors, theory in comparisons:
        ok = n_errors >= MIN_ERRORS
        rel = ser[ok] / theory[ok] - 1
        z = z_score(ser[ok], theory[ok], theory[ok] * (1 - theory[ok]) / n_symbols)
        lines.append(f"{label:>36}{ok.sum():>8}{rel.mean():>+11.1%}"
                     f"{np.abs(rel).max():>12.1%}{np.abs(z).max():>9.2f}")
    return lines


def format_gap(gap_db):
    """SNR gap as text; '-' if the SER level is not reached."""
    return "-" if np.isnan(gap_db) else f"{gap_db:.2f}"


def format_db(value_db):
    """dB value as text, printing 0.00 instead of -0.00."""
    return f"{round(value_db, 2) + 0.0:.2f}"
