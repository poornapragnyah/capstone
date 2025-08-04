import argparse
import numpy as np
import matplotlib.pyplot as plt
import mne

def plot_channel(channel_data, sfreq, title, filename):
    time = np.arange(channel_data.shape[0]) / sfreq
    plt.figure(figsize=(12, 4))
    plt.plot(time, channel_data)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Verify structure and trigger data from a .npy EEG file.")
    parser.add_argument("file_path", help="Path to the .npy file.")
    parser.add_argument("sfreq", type=float, help="Sampling frequency (Hz).")
    parser.add_argument("--trial-index", type=int, default=0,
                        help="Index of trial to load (default: 0)")
    args = parser.parse_args()

    raw_data = np.load(args.file_path)
    print(f"Loaded data shape: {raw_data.shape}")

    if raw_data.ndim != 3:
        raise ValueError(f"Expected shape (n_trials, n_channels, n_samples), but got {raw_data.shape}")

    if args.trial_index >= raw_data.shape[0]:
        raise IndexError(f"Trial index {args.trial_index} out of bounds. Max allowed is {raw_data.shape[0]-1}.")

    data = raw_data[args.trial_index]
    print(f"Using trial {args.trial_index}, shape: {data.shape} (channels, samples)")

    n_channels, n_samples = data.shape
    ch_names = [f"CH{i}" for i in range(n_channels)]

    # Try assuming last channel is stim and plot it
    stim_channel = data[-1]
    print(f"\nAssuming last channel (CH{n_channels-1}) is stimulus channel.")
    print(f"Unique values: {np.unique(stim_channel)}")
    print(f"Min: {stim_channel.min()}, Max: {stim_channel.max()}, Std: {np.std(stim_channel)}")

    plot_channel(stim_channel, args.sfreq, f"Channel CH{n_channels-1} (Assumed Stim)", "stim_channel_plot.png")

    # Optional preprocessing: round to nearest int (comment out if not needed)
    stim_channel_clean = np.round(stim_channel).astype(int)
    data[-1] = stim_channel_clean

    # Try finding events assuming last channel is stim
    ch_types = ['misc'] * (n_channels - 1) + ['stim']
    info = mne.create_info(ch_names=ch_names, sfreq=args.sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    events = mne.find_events(raw, stim_channel=ch_names[-1], shortest_event=1)
    print(f"\nEvents from CH{n_channels-1}: {len(events)} found.")
    if len(events) > 0:
        print(f"Unique event IDs: {np.unique(events[:, 2])}")
        print(f"Time between events (s): {np.diff(events[:, 0]) / args.sfreq}")

    # Brute-force check: try all channels
    print("\nTrying all channels as potential stim channels...")
    for i in range(n_channels):
        ch_types_try = ['misc'] * n_channels
        ch_types_try[i] = 'stim'
        info_try = mne.create_info(ch_names=ch_names, sfreq=args.sfreq, ch_types=ch_types_try)
        raw_try = mne.io.RawArray(data, info_try)
        try:
            events_try = mne.find_events(raw_try, stim_channel=ch_names[i], shortest_event=1)
            if len(events_try) > 0:
                print(f"✅ CH{i} could be stim channel — {len(events_try)} events found.")
        except Exception as e:
            print(f"❌ CH{i} failed: {e}")

if __name__ == "__main__":
    main()
