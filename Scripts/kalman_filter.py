import pykalman
import attr
import argparse
import operator
from typing import Callable, Deque, Dict, List, Optional, Tuple, TypeVar, Text

import numpy as np

from numpy import ma
from pykalman import KalmanFilter
from collections import defaultdict

from sleap import Labels
from sleap.instance import Track 
from sleap.nn.tracking import TrackCleaner

@attr.s(auto_attribs=True)
class TrackKalman:

    kalman_filters: List[pykalman.KalmanFilter]
    state_means: List[list]
    state_covariances: List[list]
    tracks: List[str]
    frame_tracking: List[list]
    instance_count: int

    @classmethod
    def initialize(cls, frames, instance_count):

        frame_array_dict = defaultdict(list)

        kalman_filter_list = []
        state_means_list = []
        state_covariances_list = []
        track_list = []
        frame_tracking_list = []

        for lf in frames:

            for inst in lf.instances: 

                frame_array_dict[inst.track.name].append(inst.points_array.flatten()[:6])

        for track_name, frame_array in frame_array_dict.items():

            frame_array = ma.asarray(frame_array)

            frame_array = ma.masked_invalid(frame_array)

            inital_frame = frame_array[0]

            inital_frame_size = inital_frame.size

            initial_state_means = [0] * (inital_frame_size * 2)

            for coord_pos, coord_value in enumerate(inital_frame.flatten()):

                initial_state_means[(coord_pos * 2)] = coord_value

            transition_matrix = []

            for coord_pos in range(0, inital_frame_size):

                transition_matrix.append([int(x in [(coord_pos * 2), (coord_pos * 2) + 1]) for x in range(inital_frame_size * 2)])

                transition_matrix.append([int(x == ((coord_pos * 2) + 1)) for x in range(inital_frame_size * 2)])

            observation_matrix = []

            for coord_pos in range(0, inital_frame_size):

                observation_matrix.append([int(x == (coord_pos * 2)) for x in range(inital_frame_size * 2)])

            kf = KalmanFilter(transition_matrices = transition_matrix,
                               observation_matrices = observation_matrix,
                               initial_state_mean = initial_state_means)

            kf = kf.em(frame_array, n_iter=20)

            state_means, state_covariances = kf.filter(frame_array)

            kalman_filter_list.append(kf)
            state_means_list.append(list(state_means)[-1])
            state_covariances_list.append(list(state_covariances)[-1])
            track_list.append(track_name)
            frame_tracking_list.append([track_name] * len(frames))

        return cls(kalman_filters = kalman_filter_list, 
                   state_means = state_means_list, 
                   state_covariances = state_covariances_list,
                   tracks = track_list,
                   frame_tracking = frame_tracking_list,
                   instance_count = instance_count)

    def track_frames(self, frames, instance_count):

        for frames_pos in range(len(frames)):

            if instance_count == 0 or instance_count > 2:
                raise Exception('instance_count only allowed to be 1 or 2')

            if instance_count == 1:

                smallest_distance_mean = None

                smallest_distance_track = None

                smallest_distance_points_array = None

                larger_distance_mean = []

                larger_distance_points_array = []
                
                exp_mean, exp_covariance = self.kalman_filters[0].filter_update(self.state_means[0], self.state_covariances[0], ma.masked)

                exp_coord_means = np.array([exp_mean[pos] for pos in range(0, len(exp_mean), 2)])

                for inst in frames[frames_pos].instances:

                    if inst.score <= 0.30:

                        if not smallest_distance_track:

                            smallest_distance_points_array = ma.masked

                        continue

                    weights = []

                    inst_points = []

                    for node in ['thor', 'head', 'abdo']:

                        weights.extend([inst[node].score, inst[node].score])
                        inst_points.extend([inst[node].x, inst[node].y])

                    weights = np.asarray(weights)
         
                    distances = abs(exp_coord_means - inst_points)

                    if all(np.isnan(distances)):

                        if not smallest_distance_track:

                            smallest_distance_points_array = ma.masked

                        continue

                    distances = ma.MaskedArray(distances, mask=np.isnan(distances))

                    current_distance_mean = ma.average(distances, weights=weights)

                    if smallest_distance_mean == None:

                        smallest_distance_mean = current_distance_mean

                        smallest_distance_track = inst.track.name

                        smallest_distance_points_array = inst.points_array.flatten()[:6]

                    elif current_distance_mean < smallest_distance_mean:

                        larger_distance_mean.append(smallest_distance_mean)

                        larger_distance_points_array.append(smallest_distance_points_array)

                        smallest_distance_mean = current_distance_mean

                        smallest_distance_track = inst.track.name

                        smallest_distance_points_array = inst.points_array.flatten()[:6]

                if len(larger_distance_mean) != 0:

                    for larger_distance_pos in range(larger_distance_mean):

                        point_difference = abs(smallest_distance_points_array - larger_distance_points_array[larger_distance_pos])

                        if not all(np.isnan(point_difference)):

                            point_difference_mean = np.nanmean(point_difference)

                            if point_difference_mean < smallest_distance_mean:

                                smallest_distance_points_array = ma.masked
                                smallest_distance_track = None

                if ma.is_masked(smallest_distance_points_array):

                    self.frame_tracking[0] = self.frame_tracking[0] + [None]

                else:

                    points_array = ma.asarray(smallest_distance_points_array)

                    points_array = ma.masked_invalid(points_array)

                    last_mean, last_covariance = self.kalman_filters[0].filter_update(self.state_means[0], self.state_covariances[0], points_array)

                    self.state_means[0] = last_mean
                    self.state_covariances[0] = last_covariance
                    self.frame_tracking[0] = self.frame_tracking[0] + [smallest_distance_track]

            if instance_count == 2:

                smallest_distance_mean = [None, None]

                smallest_distance_track = [None, None]

                smallest_distance_points_array = [None, None]

                for kalman_pos, kalman_filter in enumerate(self.kalman_filters):

                    exp_mean, exp_covariance = kalman_filter.filter_update(self.state_means[kalman_pos], self.state_covariances[kalman_pos], ma.masked)

                    exp_coord_means = np.array([exp_mean[pos] for pos in range(0, len(exp_mean), 2)])

                    for inst in frames[frames_pos].instances:

                        if inst.score <= 0.30:

                            if not smallest_distance_track[kalman_pos]:

                                smallest_distance_points_array[kalman_pos] = ma.masked

                            continue

                        weights = []

                        inst_points = []

                        for node in ['thor', 'head', 'abdo']:

                            weights.extend([inst[node].score, inst[node].score])
                            inst_points.extend([inst[node].x, inst[node].y])

                        weights = np.asarray(weights)
             
                        distances = abs(exp_coord_means - inst_points)

                        if all(np.isnan(distances)):

                            if not smallest_distance_track[kalman_pos]:

                                smallest_distance_points_array[kalman_pos] = ma.masked

                            continue

                        distances = ma.MaskedArray(distances, mask=np.isnan(distances))

                        current_distance_mean = ma.average(distances, weights=weights)

                        if smallest_distance_mean[kalman_pos] == None:

                            smallest_distance_mean[kalman_pos] = current_distance_mean

                            smallest_distance_track[kalman_pos] = inst.track.name

                            smallest_distance_points_array[kalman_pos] = inst.points_array.flatten()[:6]

                        elif current_distance_mean < smallest_distance_mean[kalman_pos]:

                            smallest_distance_mean[kalman_pos] = current_distance_mean

                            smallest_distance_track[kalman_pos] = inst.track.name

                            smallest_distance_points_array[kalman_pos] = inst.points_array.flatten()[:6]

                if len(frames[frames_pos].instances) == 1 and smallest_distance_track[0] and smallest_distance_track[0] == smallest_distance_track[1]:

                    inst_difference = abs(smallest_distance_mean[0] - smallest_distance_mean[1])

                    inst_function = min(smallest_distance_mean)

                    if (inst_difference / inst_function) <= 1:

                        smallest_distance_track = [None, None]
                        smallest_distance_points_array = [ma.masked, ma.masked]
                           
                if len(set(smallest_distance_track)) == 1 and smallest_distance_track[0]:

                    if smallest_distance_mean[0] < smallest_distance_mean[1]:

                        if len(frames[frames_pos].instances) == 1:

                            smallest_distance_mean[1] = None
                            smallest_distance_track[1] = None
                            smallest_distance_points_array[1] = ma.masked

                        elif len(frames[frames_pos].instances) == 2:

                            for inst in frames[frames_pos].instances:

                                if inst.track.name != smallest_distance_track[1]:

                                    smallest_distance_mean[1] = None
                                    smallest_distance_track[1] = inst.track.name
                                    smallest_distance_points_array[1] = inst.points_array.flatten()[:6]
                                    break

                        else:
                            smallest_distance_mean[1] = None
                            smallest_distance_track[1] = None
                            smallest_distance_points_array[1] = ma.masked

                    else:

                        if len(frames[frames_pos].instances) == 1:

                            smallest_distance_mean[0] = None
                            smallest_distance_track[0] = None
                            smallest_distance_points_array[0] = ma.masked

                        elif len(frames[frames_pos].instances) == 2:

                            for inst in frames[frames_pos].instances:

                                if inst.track.name != smallest_distance_track[0]:

                                    smallest_distance_mean[0] = None
                                    smallest_distance_track[0] = inst.track.name
                                    smallest_distance_points_array[0] = inst.points_array.flatten()[:6]
                                    break

                        else:

                            smallest_distance_mean[0] = None
                            smallest_distance_track[0] = None
                            smallest_distance_points_array[0] = ma.masked

                if smallest_distance_track[0] != smallest_distance_track[1] and smallest_distance_track[0] and smallest_distance_track[1] and smallest_distance_mean[0] and smallest_distance_mean[1]:

                    point_difference = abs(smallest_distance_points_array[0] - smallest_distance_points_array[1])

                    if not all(np.isnan(point_difference)):

                        point_difference_mean = np.nanmean(point_difference)

                        if point_difference_mean < smallest_distance_mean[0] and point_difference_mean < smallest_distance_mean[1]:

                            if smallest_distance_mean[0] > smallest_distance_mean[1]:

                                smallest_distance_points_array[0] = ma.masked
                                smallest_distance_track[0] = None

                            if smallest_distance_mean[1] > smallest_distance_mean[0]:

                                smallest_distance_points_array[1] = ma.masked
                                smallest_distance_track[1] = None

                for smallest_pos in range(len(smallest_distance_track)):

                    if ma.is_masked(smallest_distance_points_array[smallest_pos]):

                        self.frame_tracking[smallest_pos] = self.frame_tracking[smallest_pos] + [None]

                    else:

                        points_array = ma.asarray(smallest_distance_points_array[smallest_pos])

                        points_array = ma.masked_invalid(points_array)

                        last_mean, last_covariance = self.kalman_filters[smallest_pos].filter_update(self.state_means[smallest_pos], self.state_covariances[smallest_pos], points_array)

                        self.state_means[smallest_pos] = last_mean
                        self.state_covariances[smallest_pos] = last_covariance
                        self.frame_tracking[smallest_pos] = self.frame_tracking[smallest_pos] + [smallest_distance_track[smallest_pos]]

    def get_tracking_array (self):

        return np.asarray(self.frame_tracking)

def filter_frames (frames, instance_count, keep_non_tracked = False, init_len = 10):

    # Initialize the filter
    kalman_filter = TrackKalman.initialize(frames[:init_len], instance_count)

    # Run the filter, frame by frame
    kalman_filter.track_frames(frames[init_len:], instance_count)

    # Assign the tracking array
    tracking_array = kalman_filter.get_tracking_array()

    # Create list to store the initial tracks
    initial_tracks = [None] * tracking_array.shape[0]

    # Loop the frames
    for frame_pos, lf in enumerate(frames):

        # Assign the predicted tracks
        predicted_track_names = tracking_array[:,frame_pos]

        # Check if this is the first frame
        if frame_pos == 0:

            # Loop the frame tracking data
            for track_pos in range(len(predicted_track_names)):

                # Assign the initial track name
                initial_track_name = predicted_track_names[track_pos]

                # Loop the instances
                for inst in lf.instances:

                    # Check if current instance is the correct initial track
                    if inst.track.name == initial_track_name:

                        initial_tracks[track_pos] = inst.track

        # Create list of instances that were not tracked
        non_tracked_instances = []

        # Loop the instances
        for inst in lf.instances:

            # Check if current instances is on the predicted track
            if inst.track.name not in predicted_track_names:

                # Append the instance to remove
                non_tracked_instances.append(inst)

        # Check if non-tracked instances should be kept
        if keep_non_tracked:

            # Loop the instances to remove
            for non_tracked_pos, non_tracked_instance in enumerate(non_tracked_instances):

                new_untracked_track = Track(spawned_on = frame_pos, name = f"untracked_{frame_pos}_{non_tracked_pos}")

                non_tracked_instance.track = new_untracked_track

        # Otherwise remove the non-tracked instances
        else:

            # Loop the instances to remove
            for non_tracked_instance in non_tracked_instances:

                # Remove the instance
                lf.instances.remove(non_tracked_instance)

        # Loop the instances
        for inst in lf.instances:

            # Loop the frame tracking data
            for initial_track, predicted_track_name in zip(initial_tracks, predicted_track_names):

                # Check if current instances is on the predicted track
                if inst.track.name == predicted_track_name:

                    # Update the track assignment
                    inst.track = initial_track
                    
                    # Break if assigned, to avoid another assignment
                    break

def fit_tracks (filename: Text, instance_count: int, keep_non_tracked: bool, iou_threshold: float):
    """Wraps `TrackCleaner` for easier cli api."""

    labels = Labels.load_file(filename)
    video = labels.videos[0]
    frames = labels.find(video)

    TrackCleaner(instance_count=instance_count, iou_threshold=iou_threshold).run(frames=frames)

    filter_frames(frames=frames, instance_count=instance_count, keep_non_tracked=keep_non_tracked)

    # Rebuild list of tracks
    labels.tracks = list(
        {
            instance.track
            for frame in labels
            for instance in frame.instances
            if instance.track
        }
    )

    labels.tracks.sort(key=operator.attrgetter("spawned_on", "name"))

    # Save new file
    save_filename = filename
    save_filename = save_filename.replace(".slp", ".cleaned.slp")
    save_filename = save_filename.replace(".h5", ".cleaned.h5")
    save_filename = save_filename.replace(".json", ".cleaned.json")

    Labels.save_file(labels, save_filename)

    print(f"Saved: {save_filename}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    parser.add_argument("-c",
                        "--instance_count",
                        type=int,
                        default=2,
                        help="Count of instances to keep in each frame")
    parser.add_argument("-k",
                        "--keep-non-tracked",
                        action='store_true')
    parser.add_argument("-i",
                        "--iou_threshold",
                        type=float)

    args = parser.parse_args()

    fit_tracks(filename=args.data_path, instance_count=args.instance_count, keep_non_tracked=args.keep_non_tracked, iou_threshold=args.iou_threshold)