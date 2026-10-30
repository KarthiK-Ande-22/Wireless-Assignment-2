"""Read quantities off simulated SER curves: SNR at a given SER, slopes, deviations."""
import numpy as np

from .plotting import MIN_ERRORS


def snr_at_ser(snr_db, ser, n_errors, target):
    """SNR (dB) at which a simulated SER curve crosses `target`; NaN if not reached."""
    reliable = n_errors >= MIN_ERRORS                            # same points as plotted
    log_ser, snr = np.log10(ser[reliable]), snr_db[reliable]
    if not log_ser.min() <= np.log10(target) <= log_ser.max():   # target outside the curve
        return np.nan
    return np.interp(-np.log10(target), -log_ser, snr)           # linear in (SNR_dB, log10 SER)


def theory_snr_at_ser(ser_function, target, snr_db_max=80.0):
    """SNR (dB) at which a theoretical SER curve `ser_function(snr_linear)` crosses `target`."""
    snr_db = np.arange(-10.0, snr_db_max, 0.01)                  # fine grid; log10 SER is smooth in dB
    ser = ser_function(10 ** (snr_db / 10))
    ok = ser > 0                                                 # AWGN SER underflows to 0 at high SNR
    return np.interp(-np.log10(target), -np.log10(ser[ok]), snr_db[ok])   # SER decreases with SNR


def high_snr_slopes(snr_db, ser, n_errors, n_segments=3, min_errors=50):
    """Local slopes -d log10(SER) / d log10(SNR) over the last steps of a curve.

    At high SNR the slope tends to the diversity order. Returns the slopes of the
    last `n_segments` steps between points with >= `min_errors` errors, and the
    SNR values (dB) of those points.
    """
    reliable = n_errors >= min_errors                            # 50 errors: accurate to ~15 %
    snr = snr_db[reliable][-(n_segments + 1):]                   # high-SNR end of the curve
    log_ser = np.log10(ser[reliable][-(n_segments + 1):])
    return -np.diff(log_ser) / np.diff(snr / 10), snr            # decades of SER per decade of SNR


def loglog_slope(snr_db, ser):
    """Least-squares slope -d log10(SER) / d log10(SNR) through the given points."""
    return -np.polyfit(np.asarray(snr_db) / 10, np.log10(ser), 1)[0]   # straight line in log-log


def fitted_slope(snr_db, ser, n_errors, snr_min_db):
    """Least-squares slope -d log10(SER) / d log10(SNR) over the plotted points with SNR >= snr_min_db."""
    ok = (n_errors >= MIN_ERRORS) & (snr_db >= snr_min_db)          # high-SNR, plotted points
    return loglog_slope(snr_db[ok], ser[ok])


def z_score(p_a, p_b, variance):
    """Difference of two SER values in units of its Monte Carlo standard deviation."""
    return (p_a - p_b) / np.sqrt(variance)


def agreement_lines(comparisons, n_symbols):
    """Text table of how closely simulated curves follow theory, over the plotted points.

    `comparisons` holds (label, SER estimates, error counts, theoretical SER). Each row
    gives the number of points with >= MIN_ERRORS errors, the mean and the largest
    relative deviation, and the largest deviation in units of sigma = sqrt(P (1 - P) / N).
    """
    lines = [f"Agreement with theory over the plotted points (>= {MIN_ERRORS} errors), "
             "sigma = sqrt(P (1 - P) / N)",
             f"{'simulation vs theory':>36}{'points':>8}{'mean dev.':>11}{'max |dev.|':>12}{'max |z|':>9}"]
    for label, ser, n_errors, theory in comparisons:
        ok = n_errors >= MIN_ERRORS                                   # plotted points only
        rel = ser[ok] / theory[ok] - 1                                # relative deviation
        z = z_score(ser[ok], theory[ok], theory[ok] * (1 - theory[ok]) / n_symbols)
        lines.append(f"{label:>36}{ok.sum():>8}{rel.mean():>+11.1%}"
                     f"{np.abs(rel).max():>12.1%}{np.abs(z).max():>9.2f}")
    return lines


def format_gap(gap_db):
    """Format an SNR gap in dB; '-' when the SER level is not reached by both curves."""
    return "-" if np.isnan(gap_db) else f"{gap_db:.2f}"


def format_db(value_db):
    """Format a dB value with two decimals, printing 0.00 rather than -0.00."""
    return f"{round(value_db, 2) + 0.0:.2f}"                     # + 0.0 turns -0.0 into 0.0
