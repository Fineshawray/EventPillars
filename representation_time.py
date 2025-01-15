import torch
from torch_scatter import scatter, scatter_max, scatter_min
import numpy as np
from src.io.psee_loader import PSEELoader
import time
from tqdm import tqdm
import os
from dataset_encoder.hats import HATS
from typing import Union, Tuple
import torch.nn as nn


class Operations(object):
    def __init__(self, func, height, width):
        self.func = func
        self.height = height
        self.width = width

    def __call__(self, events):
        return self.exec(events)

    def run(self, src, index, aggregation):
        event_surface = scatter(
            src=src,
            index=index,
            dim_size=self.height * self.width,
            reduce=aggregation,
        )
        event_surface = event_surface.reshape(self.height, self.width)

        return event_surface

    def exec(self, events):
        index = torch.tensor(events[:, 0] + events[:, 1] * self.width).to(torch.int64)

        if self.func == "DiST":
            CLIP_COUNT_RATE = 0.99
            DISC_ALPHA = 3.0
            H = self.height
            W = self.width
            pos = events[events[:, 3] == 1]
            neg = events[events[:, 3] == -1]
            pos = torch.from_numpy(pos)
            neg = torch.from_numpy(neg)
            # Get pos, neg counts
            pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
            pos_count = pos_count.float()

            neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)
            neg_count = neg_count.float()

            # clip count
            pos_unique_count = torch.unique(pos_count, return_counts=True)[1]
            pos_sum_subset = torch.cumsum(pos_unique_count, dim=0)
            pos_th_clip = pos_sum_subset[pos_sum_subset < H * W * CLIP_COUNT_RATE].shape[0]
            pos_count[pos_count > pos_th_clip] = pos_th_clip

            neg_unique_count = torch.unique(neg_count, return_counts=True)[1]
            neg_sum_subset = torch.cumsum(neg_unique_count, dim=0)
            neg_th_clip = neg_sum_subset[neg_sum_subset < H * W * CLIP_COUNT_RATE].shape[0]
            neg_count[neg_count > neg_th_clip] = neg_th_clip

            start_time = events[0, 2]
            time_length = events[-1, 2] - events[0, 2]

            norm_pos_time = (pos[:, 2] - start_time) / time_length
            norm_neg_time = (neg[:, 2] - start_time) / time_length

            # Get pos, neg time
            pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
            neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
            pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
            pos_min_out, _ = scatter_min(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
            pos_out = pos_out.reshape(H, W).float()
            pos_min_out = pos_min_out.reshape(H, W).float()
            neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
            neg_min_out, _ = scatter_min(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
            neg_out = neg_out.reshape(H, W).float()
            neg_min_out = neg_min_out.reshape(H, W).float()

            pos_min_out[pos_count == 0] = 1.0
            neg_min_out[neg_count == 0] = 1.0

            # Get temporal discount
            pos_disc = torch.zeros_like(pos_count)
            neg_disc = torch.zeros_like(neg_count)

            patch_size = 5

            pos_neighbor_count = patch_size ** 2 * torch.nn.functional.avg_pool2d(pos_count.unsqueeze(0), patch_size,
                                                                                  stride=1,
                                                                                  padding=patch_size // 2)
            neg_neighbor_count = patch_size ** 2 * torch.nn.functional.avg_pool2d(neg_count.unsqueeze(0), patch_size,
                                                                                  stride=1,
                                                                                  padding=patch_size // 2)

            pos_disc = (torch.nn.functional.max_pool2d(pos_out.unsqueeze(0), patch_size, stride=1,
                                                       padding=patch_size // 2) +
                        torch.nn.functional.max_pool2d(-pos_min_out.unsqueeze(0), patch_size, stride=1,
                                                       padding=patch_size // 2)) / \
                       (pos_neighbor_count)
            neg_disc = (torch.nn.functional.max_pool2d(neg_out.unsqueeze(0), patch_size, stride=1,
                                                       padding=patch_size // 2) +
                        torch.nn.functional.max_pool2d(-neg_min_out.unsqueeze(0), patch_size, stride=1,
                                                       padding=patch_size // 2)) / \
                       (neg_neighbor_count)

            pos_out[pos_count > 0] = (pos_out[pos_count > 0] - DISC_ALPHA * pos_disc.squeeze()[pos_count > 0])
            pos_out[pos_out < 0] = 0
            pos_out[pos_neighbor_count.squeeze() == 1.0] = 0
            neg_out[neg_count > 0] = (neg_out[neg_count > 0] - DISC_ALPHA * neg_disc.squeeze()[neg_count > 0])
            neg_out[neg_out < 0] = 0
            neg_out[neg_neighbor_count.squeeze() == 1.0] = 0

            pos_out = pos_out.reshape(H * W)
            neg_out = neg_out.reshape(H * W)

            pos_val, pos_idx = torch.sort(pos_out)
            neg_val, neg_idx = torch.sort(neg_out)

            pos_unq, pos_cnt = torch.unique_consecutive(pos_val, return_counts=True)
            neg_unq, neg_cnt = torch.unique_consecutive(neg_val, return_counts=True)

            pos_sort = torch.zeros_like(pos_out)
            neg_sort = torch.zeros_like(neg_out)

            pos_sort[pos_idx] = torch.repeat_interleave(torch.arange(pos_unq.shape[0]), pos_cnt).float() / \
                                pos_unq.shape[0]
            neg_sort[neg_idx] = torch.repeat_interleave(torch.arange(neg_unq.shape[0]), neg_cnt).float() / \
                                neg_unq.shape[0]

            pos_sort = pos_sort.reshape(H, W)
            neg_sort = neg_sort.reshape(H, W)

            result = torch.stack([pos_sort, neg_sort], dim=2)

            result = result.permute(2, 0, 1)
            event_surface = result.float()
        elif self.func == "SAE":
            t_sequence = events[:, 2] - events[:, 2][0]
            t_sequence = torch.tensor(t_sequence)
            tmax_values = (t_sequence * 255).clamp(0, 255).byte()
            Tmax_out, _ = scatter_max(tmax_values, index, dim=-1, dim_size=self.height * self.width)
            Tmax = Tmax_out.reshape(self.height, self.width)
            result = Tmax
            event_surface = result.float()
        elif self.func == "SAE_UAE":
            t_sequence = events[:, 2] - events[:, 2][0]
            t_sequence = torch.tensor(t_sequence)
            t_values = (t_sequence * 255).clamp(0, 255).byte()

            # Tmax
            Tmax_out, _ = scatter_max(t_values, index, dim=-1, dim_size=self.height * self.width)
            Tmax = Tmax_out.reshape(self.height, self.width)

            # Tmin
            Tmin_out, _ = scatter_min(t_values, index, dim=-1, dim_size=self.height * self.width)
            Tmin = Tmin_out.reshape(self.height, self.width)

            result = torch.stack([Tmax, Tmin], dim=2)

            event_surface = result.float()
        elif self.func == "SAE_UAE_AEP":
            pos = events[events[:, 3] == 1]
            neg = events[events[:, 3] == -1]
            pos = torch.from_numpy(pos)
            neg = torch.from_numpy(neg)

            Histogram_neg = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * self.width,
                                           minlength=self.height * self.width).reshape(self.height, self.width)
            Histogram_pos = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * self.width,
                                           minlength=self.height * self.width).reshape(self.height, self.width)

            t_sequence = events[:, 2] - events[:, 2][0]
            t_sequence = torch.tensor(t_sequence)
            t_values = (t_sequence * 255).clamp(0, 255).byte()

            # Tmax
            Tmax_out, _ = scatter_max(t_values, index, dim=-1, dim_size=self.height * self.width)
            Tmax = Tmax_out.reshape(self.height, self.width)

            # Tmin
            Tmin_out, _ = scatter_min(t_values, index, dim=-1, dim_size=self.height * self.width)
            Tmin = Tmin_out.reshape(self.height, self.width)

            tanh = nn.Tanh()
            AEP = tanh(Histogram_neg - Histogram_pos)
            result = torch.stack([Tmax, Tmin, AEP], dim=2)

            event_surface = result.float()
        elif self.func == "SAE_UAE_AEP_ED" or "EventPillar":
            pos = events[events[:, 3] == 1]
            neg = events[events[:, 3] == -1]
            pos = torch.from_numpy(pos)
            neg = torch.from_numpy(neg)

            Histogram_neg = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * self.width,
                                           minlength=self.height * self.width).reshape(self.height, self.width)
            Histogram_pos = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * self.width,
                                           minlength=self.height * self.width).reshape(self.height, self.width)

            t_sequence = events[:, 2] - events[:, 2][0]
            t_sequence = torch.tensor(t_sequence)
            t_values = (t_sequence * 255).clamp(0, 255).byte()

            # Tmax
            Tmax_out, _ = scatter_max(t_values, index, dim=-1, dim_size=self.height * self.width)
            Tmax = Tmax_out.reshape(self.height, self.width)

            # Tmin
            Tmin_out, _ = scatter_min(t_values, index, dim=-1, dim_size=self.height * self.width)
            Tmin = Tmin_out.reshape(self.height, self.width)

            tanh = nn.Tanh()
            AEP = tanh(Histogram_neg - Histogram_pos)
            num_events = Histogram_pos + Histogram_neg
            ED = num_events / num_events.max()
            result = torch.stack([Tmax, Tmin, AEP, ED], dim=2)

            event_surface = result.float()
        return event_surface.numpy()


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols = 2 * pols.astype(np.int8) - 1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


class ToTimesurface:
    """Create global or local time surfaces for each event. Modeled after the paper Lagorce et al.
    2016, Hots: a hierarchy of event-based time-surfaces for pattern recognition
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476.
    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
    """
    surface_dimensions: Union[None, Tuple[int, int]] = None
    tau: float = 5e3
    decay: str = "lin"

    def __init__(self, sensor_size):
        self.sensor_size = sensor_size

    def __call__(self, events, indices):
        timestamp_memory = np.zeros(
            (self.sensor_size[2], self.sensor_size[1], self.sensor_size[0])
        )
        timestamp_memory -= self.tau * 3 + 1
        all_surfaces = np.zeros(
            (
                len(indices),
                self.sensor_size[2],
                self.sensor_size[1],
                self.sensor_size[0],
            )
        )

        to_timesurface_numpy(
            events["x"],
            events["y"],
            events["t"],
            events["p"],
            indices,
            timestamp_memory,
            all_surfaces,
            tau=self.tau,
        )
        return all_surfaces


def to_timesurface_numpy(x, y, t, p, indices, timestamp_memory, all_surfaces, tau=5e3):
    """Representation that creates timesurfaces for each event in the recording. Modeled after the
    paper Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern
    recognition https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476.
    Parameters:
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
    Returns:
        array of timesurfaces with dimensions (w,h) or (p,w,h)
    """
    current_index_pos = 0
    for index in range(len(x)):
        timestamp_memory[p[index], y[index], x[index]] = t[index]

        if index == indices[current_index_pos]:
            timestamp_context = timestamp_memory - t[index]
            all_surfaces[current_index_pos, :, :, :] = np.exp(timestamp_context / tau)
            current_index_pos += 1
            if current_index_pos > len(indices) - 1:
                break


def encode_events(representation):
    # DiST, EventPillar, HOTS, HATS, Voxel Grid
    phases = ['n-cars_test', 'n-cars_train']
    classes = ['background', 'cars']
    dataset_folders = [ph for ph in phases]
    total_time = 0
    for dataset_folder in dataset_folders:
        for cls in classes:
            src_dat_folder = os.path.join('dataset/n_cars', dataset_folder, cls)

            for dat in tqdm([i for i in os.listdir(src_dat_folder) if '.dat' in i]):

                video = PSEELoader(os.path.join(src_dat_folder, dat))
                cur_events = video.load_delta_t(100000)
                x, y, t, p = cur_events['x'], cur_events['y'], cur_events['t'], cur_events['p']
                t = t - t.min()
                time_interval = t.max() - t.min()
                t_s = t / time_interval
                p = 2 * p.astype(np.int8) - 1
                events = np.concatenate(
                    [
                        x[..., np.newaxis],
                        y[..., np.newaxis],
                        t_s[..., np.newaxis],
                        p[..., np.newaxis],
                    ],
                    axis=1,
                )
                start_time = time.time()
                if representation == "DiST":
                    encoded_event = Operations(func="hyper_histogram", height=y.max() + 1, width=x.max() + 1)(events)
                elif representation == "Event_Pillar":
                    encoded_event = Operations(func="Event_Pillar", height=y.max() + 1, width=x.max() + 1)(events)
                elif representation == "SAE":
                    encoded_event = Operations(func="SAE", height=y.max() + 1, width=x.max() + 1)(events)
                elif representation == "HATS":
                    hats = HATS(temp_window=1, width=max(cur_events['x']) + 1, height=max(cur_events['y']) + 1)
                    encoded_event = hats.process_all(cur_events)
                elif representation == "Voxel_Grid":
                    encoded_event = events_to_voxel_grid(events, 9, x.max() + 1, y.max() + 1)
                elif representation == "HOTS":
                    time_surface = ToTimesurface(sensor_size=(cur_events["x"].max() + 1, cur_events["y"].max() + 1, 2))
                    all_indices = np.arange(len(cur_events["x"]))
                    all_surfaces = time_surface(cur_events, all_indices)
                end_time = time.time()
                file_time = end_time - start_time
                total_time += file_time

    time_per_sample = total_time/24029*1000
    print(f"Total time for encoding: {total_time:.4f} seconds")
    print(f"time for encoding one sample: {time_per_sample:.4f} milliseconds")


def main():
    # Specify the encoding method you want to use
    encoding_method = "DiST"  # Change this to "EventPillar", "SAE", "SAE_UAE", "SAE_UAE_AEP","SAE_UAE_AEP_ED",
    # "HATS", "Voxel_Grid", or "HOTS"
    encode_events(encoding_method)


if __name__ == "__main__":
    main()
