import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import scipy.interpolate as interp

VIDEO_DURATION = 240 # in seconds

# Directories
folder_dir = '/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints'

# Joint groups
JOINT_GROUPS = {
    0: [3, 12, 15],          # head / neck
    1: [1, 2, 6, 9],         # hips / lower body
    2: [16, 17, 18, 19],     # shoulders / elbows
    3: [20, 21, 22, 23]      # wrists / hands
}

# Cubic spline interpolation
def spline_interp(sig):
    sig = sig.copy()
    t = np.arange(len(sig))
    good = ~np.isnan(sig)

    if good.sum() < 4:
        return sig

    f = interp.interp1d(
        t[good],
        sig[good],
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )

    sig[~good] = f(t[~good])
    return sig


# PSD-based cutoff estimation
def determine_initial_cutoffs_from_data(raw_data, joint_groups, fs, power_threshold=0.95):

    initial_cutoffs = {}

    for group_id, joint_indices in joint_groups.items():
        cutoffs = []

        for joint_idx in joint_indices:
            for coord in [0, 1]:
                sig = raw_data[joint_idx, coord, :]
                sig = spline_interp(sig)

                if np.isnan(sig).all():
                    continue

                nperseg = min(256, len(sig))
                freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)

                df = freqs[1] - freqs[0]
                cumulative_power = np.cumsum(psd) * df
                total_power = cumulative_power[-1]

                cutoff_idx = np.searchsorted(
                    cumulative_power, power_threshold * total_power
                )
                cutoff = freqs[min(cutoff_idx, len(freqs) - 1)]
                cutoffs.append(cutoff)

        initial_cutoffs[group_id] = (
            np.median(cutoffs) if len(cutoffs) > 0 else np.nan
        )

        print(
            f"Group {group_id}: "
            f"Median cutoff = {initial_cutoffs[group_id]:.2f} Hz "
            f"(range {np.min(cutoffs):.2f}–{np.max(cutoffs):.2f} Hz)"
        )

    return initial_cutoffs


def main():

    # Store cutoffs across ALL videos
    all_video_cutoffs = {
        'infant': {gid: [] for gid in JOINT_GROUPS},
        'parent': {gid: [] for gid in JOINT_GROUPS}
    }

    for fname in os.listdir(folder_dir):
        if not fname.endswith('.mat'):
            continue

        print(f"\nProcessing {fname}")
        data = sio.loadmat(os.path.join(folder_dir, fname))

        labels = []
        keypoints_all = []
        head_y_values = []

        for label, person in data.items():
            if not label.startswith('person'):
                continue
            if label == 'person_2_2d':
                continue

            kp = np.array(person)
            labels.append(label)
            keypoints_all.append(kp)

            head_y = np.nan
            for f in range(kp.shape[2]):
                y = kp[15, 1, f]
                if not np.isnan(y):
                    head_y = y
                    break

            head_y_values.append(head_y)

        if len(head_y_values) < 2:
            print("  Skipping file (not enough people)")
            continue

        infant_index = int(np.nanargmax(head_y_values))
        parent_index = int(np.nanargmin(head_y_values))

        dyad = {
            'infant': keypoints_all[infant_index],
            'parent': keypoints_all[parent_index]
        }
        
        fs = dyad['parent'].shape[2]/VIDEO_DURATION

        for label, subject in dyad.items():
            print(f"\nRunning PSD analysis for {label}")
            cutoffs = determine_initial_cutoffs_from_data(
                subject, JOINT_GROUPS, fs
            )

            for gid, cutoff in cutoffs.items():
                if not np.isnan(cutoff):
                    all_video_cutoffs[label][gid].append(cutoff)


    # Average cutoff across all videos in dataset
    print("\n================ GLOBAL AVERAGE CUTOFFS ================")

    for label in ['infant', 'parent']:
        print(f"\n{label.upper()}")
        for gid, values in all_video_cutoffs[label].items():
            if len(values) == 0:
                print(f"  Group {gid}: no data")
            else:
                print(
                    f"  Group {gid}: "
                    f"mean = {np.mean(values):.2f} Hz, "
                    f"median = {np.median(values):.2f} Hz "
                    f"(n={len(values)})"
                )


if __name__ == "__main__":
    main()




