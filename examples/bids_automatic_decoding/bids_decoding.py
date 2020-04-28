from os.path import isdir
from os.path import join as opj
import copy
import itertools
import numpy as np
import pandas as pd
from pandas import read_csv
import nibabel as nib
from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel
from nilearn.decoding import Decoder
from sklearn.utils import Bunch
from joblib import Parallel, delayed


def demo_datasets(dataset):
    '''only intended for API design. Tested only for provided datasets (on drago)
    with fmriprep folder specification. It should be replaced by automatic
    bids fetching tool'''

    if dataset == "ds000105":
        t_r = 2.5
        subjects = ["{}".format(x) for x in range(1, 5)]
        task = "objectviewing"
        n_runs = 12
        conditions = ['scissors', 'face', 'cat', 'shoe',
                      'house', 'scrambledpix', 'bottle', 'chair']

        data_dir = "/storage/store/data/OpenNeuro/{}/".format(dataset)
        source_dir = opj(data_dir, "{}_R2.0.2/uncompressed/".format(dataset))

    elif dataset == "ds000107":
        # Betamaps on blocks, several per subject per condition. Use it
        t_r = 3.0
        subjects = ["{:0>2d}".format(x) for x in range(1, 5)]
        task = "onebacktask"
        n_runs = 2
        conditions = ['Words', 'Objects',
                      'Scrambled objects', 'Consonant strings']

        data_dir = "/storage/store/data/OpenNeuro/{}/".format(dataset)
        source_dir = opj(data_dir, "{}_R2.0.2/uncompressed/".format(dataset))

    elif dataset == "ds000117":
        t_r = 2.0
        subjects = ["{:0>2d}".format(x) for x in range(1, 5)]
        task = "facerecognition"
        n_runs = 9
        conditions = ["FAMOUS", "UNFAMILIAR", "SCRAMBLED"]

        data_dir = "/storage/store/data/OpenNeuro/{}/".format(dataset)
        source_dir = opj(data_dir, "{}_R1.0.3".format(dataset))

    runs = ["{:0>2d}".format(x) for x in range(1, n_runs)]
    derivatives_dir = opj(data_dir, "fmriprep/")
    out_dir = opj("/storage/store/derivatives", dataset)

    datasets_infos = Bunch(t_r=t_r, subjects=subjects, task=task, runs=runs,
                           conditions=conditions, source_dir=source_dir,
                           derivatives_dir=derivatives_dir, out_dir=out_dir)

    return datasets_infos


"""
dataset = "ds000117"
sub, run = datasets_infos.subjects[0], datasets_infos.runs[0]
(isdir(opj(datasets_infos.source_dir, "sub-{}".format(sub), "func")) is False)
(isdir(opj(datasets_infos.source_dir, "ses-mri", "sub-{}".format(sub), "func")) is True)
opj(datasets_infos.source_dir, "sub-{}".format(sub), "ses-mri", "func")

"""


def handle_non_bids_ds117(paradigm):
    if 'trial_type' not in paradigm:
        paradigm.rename(columns={"stim_type": "trial_type"}, inplace=True)
    return paradigm


def mocked_bids_fetcher(dataset):
    '''only intended for API design. Tested only for provided datasets (on drago)
    with fmriprep folder specification. It should be replaced by automatic
    bids fetching tool'''

    datasets_infos = demo_datasets(dataset)
    preprocessed_fmri, events, confounds = [], [], []
    for sub, run in itertools.product(datasets_infos.subjects,
                                      datasets_infos.runs):

        # ugly automatic handling of multimodal dataset to focus on fMRI
        if not isdir(opj(datasets_infos.source_dir, "sub-{}".format(sub), "func")) and isdir(opj(datasets_infos.source_dir, "sub-{}".format(sub), "ses-mri", "func")):
            file = opj("sub-{}".format(sub), "ses-mri", "func",
                       "sub-{}_ses-mri_task-{}_run-{}".format(sub,
                                                              datasets_infos.task, run))
        else:
            file = opj("sub-{}".format(sub), "func",
                       "sub-{}_task-{}_run-{}".format(sub, datasets_infos.task, run))

        events.append(opj(datasets_infos.source_dir,
                          "{}_events.tsv".format(file)))
        confounds.append(opj(datasets_infos.derivatives_dir,
                             "{}_bold_confounds.tsv".format(file)))
        preprocessed_fmri.append(opj(datasets_infos.derivatives_dir,
                                     "{}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz".format(file)))

    return datasets_infos, preprocessed_fmri, events, confounds


def read_clean_paradigm(event, trials_of_interest, trials_to_ignore,
                        dropna=True, verbose=0):
    """ loads events file, keep what is interesting for the user
    # COMMENT Maybe drop nans in events automatically only if they are in
    #   onset or trial_type columns.
    #
    # TO IMPLEMENT
    #   - Filtering trials_to_ignore
    #   - Trials merging, if trials of interest is a dictionary : {cond_1:
    #      [trial_a, trial_b], cond_2: [trial_c]}, all scans
    #      with labels in [trial_a, trial_b] are attributed label cond_1.
    #       Then they are handled with respect to type_of_modeling
    #       (separated_events, blocks..)
    """
    paradigm = handle_non_bids_ds117(
        read_csv(event, delimiter='\t'))
    if dropna:
        paradigm = paradigm.dropna().reset_index()
    paradigm["trial_type"] = paradigm["trial_type"].str.lower()
    paradigm['trial_type'] = paradigm["trial_type"].str.split(' ')
    paradigm["trial_type"] = paradigm["trial_type"].str.join('_')
    if verbose:
        print(np.unique(paradigm.trial_type.values, return_counts=True))
    return paradigm


def _load_confounds(confound):
    """load motion regressors from fmriprep confound.tsv file
    """
    # could handle the case of other confounds (not fmriprep) or no confounds
    reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    regs = pd.read_csv(confound, delimiter='\t').filter(
        ["X", "Y", "Z", "RotX", "RotY", "RotZ"]).fillna(0).to_numpy()
    return regs, reg_names


def make_block_paradigm(paradigm, type_of_modeling, time_between_blocks=10000):
    """
    create a suitable paradigm structure to regress events block by block

    checks two conditions to separate blocks, to be in the same block two
    events must have the same trial_type and have onset less than
    time_between_blocks otherwise, group_counter is increased.
    Events whose type is not in events_of_interest are not grouped.
    """

    paradigm['group_counter'] = ((np.abs(paradigm.onset.shift().fillna(
        0) - paradigm.onset) > time_between_blocks) |
        (paradigm.trial_type.shift() != paradigm.trial_type)).cumsum()

    # TODO if conditions not of interest, they should keep their initial label
    paradigm['trial_type'] = paradigm.trial_type + \
        "_" + paradigm.group_counter.map(str).str.zfill(2)

    return paradigm


def expand_trial_type_per_event(trial_type, per_event_type):
    """
    trial_type is the "trial_type" column of events.tsv
    per_event_type is the instance "x" of trial_type

    expands all instances of 'x' into "x_01", "x_02"..
    """
    expanded_trial_type = copy.deepcopy(trial_type)
    n_conds = (trial_type == per_event_type).sum()
    expanded_trial_type[trial_type == per_event_type] = ['%s_%0{}d'.format(
        len(str(n_conds))) % (per_event_type, i) for i in range(n_conds)]
    return expanded_trial_type


def make_event_paradigm(paradigm, events_of_interest):
    """create a suitable paradigm structure to regress events trial by trial
    """
    trial_type = paradigm["trial_type"].values
    for event_type in events_of_interest:
        trial_type = expand_trial_type_per_event(trial_type, event_type)
    return paradigm


def fit_design_matrix(fmri, model, save_location, design_matrix, event,
                      confound, type_of_modeling, trials_of_interest,
                      trials_to_ignore, hrf_model, time_between_blocks,
                      verbose):
    """ This functions create a suitable design matrix if not provided, fit it
    and save the output zmaps

    fmri : one run of one task for one subject (usually preprocessed)
    model : nilearn.stats.FirstLevelModel to fit to fmri using design_matrix
    design_matrix : matrix used to fit the model to fmri. If provided,
        all following arguments will be ignored : event, confound,
        type_of_modeling, trials_of_interest, trials_to_ignore, hrf_model
    event : event file corresponding to fmri
    confound : movement confounders to regress out (only fmriprep for now)
    type_of_modeling : string
        type_of_modeling can be in ["event-related", "block-design",
        "mumford", "session"]
    trials_of_interest : list of values of trial_type column to keep
        proposed feature if trials of interest is a
        dictionary {cond_1:[trial_a,trial_b], cond_2:[trial_c]} we could easily
    trials_to_ignore : list of values of trial_type column to filter out
    time_between_blocks : additional parameter for block-design
    Note : If some trial are not of interest nor to ignore, they will be
            regressed during GLM but then not used for decoding.
    Note 2 : By default all trial types are of interest, None to ignore
    """

    if design_matrix is None:
        n_scans = nib.load(fmri).shape[3]
        frame_times = np.arange(n_scans) * model.t_r

        if trials_of_interest == "all":
            paradigm = handle_non_bids_ds117(
                read_csv(event, delimiter='\t'))
            trials_of_interest = np.unique(paradigm.trial_type.dropna().values)
        trials_of_interest_ = ['_'.join(event.lower().split(" "))
                               for event in trials_of_interest]

        paradigm = read_clean_paradigm(event, trials_of_interest_,
                                       trials_to_ignore, verbose=verbose)

        if type_of_modeling in ["event-related", "mumford"]:
            paradigm = make_event_paradigm(paradigm, trials_of_interest_)
        elif type_of_modeling == "block-design":
            paradigm = make_block_paradigm(
                paradigm, trials_of_interest_, time_between_blocks)
        elif type_of_modeling == "session":
            paradigm = paradigm

        regs, reg_names = _load_confounds(confound)

        design_matrix = make_first_level_design_matrix(
            frame_times, events=paradigm, hrf_model=hrf_model, add_regs=regs,
            add_reg_names=reg_names)
    else:
        trials_of_interest_ = trials_of_interest
        type_of_modeling = "custom"

    model.fit(fmri, design_matrices=[design_matrix])
    # TO IMPROVE, for saving purpose for now.
    run = fmri.split('/')[-1].split('_')[-4]
    sub = fmri.split('/')[-1].split('_')[0]

    filenames = []
    for trial in design_matrix.loc[:, design_matrix.columns.str.contains('|'.join(trials_of_interest_))].columns:
        image = model.compute_contrast(design_matrix.columns == trial)
        filename = os.path.join(save_location, "{}_{}_{}_{}.nii.gz".format(
            sub, run, trial, type_of_modeling))
        image.to_filename(filename)
        filenames.append(filename)
    sorted(filenames)
    return filenames


class InterSubjectPipelineGLMDecoding():
    def __init__(self, dataset, smoothing_fwhm=5, mask=None, high_pass=.01,
                 type_of_modeling="event-r", trials_of_interest="all",
                 trials_to_ignore=[], hrf_model="spm", time_between_blocks=10000,
                 decoder=Decoder(), n_jobs=1, verbose=0):
        '''
        type_of_modeling : string
            type_of_modeling can be in ["event-related", "block-design",
            "mumford", "session"]
        trials_of_interest : list of values of trial_type column to keep
            proposed feature if trials of interest is a
            dictionary {cond_1:[trial_a,trial_b], cond_2:[trial_c]}
            trial_a,trial_b are merged into cond_1 event types
        trials_to_ignore : list of values of trial_type column to filter out
        time_between_blocks : additional parameter for block-design,
        by default, too big to be used to separate blocks.
        '''
        dataset_infos, niimgs, events, confounds = mocked_bids_fetcher(
            dataset)
        self.dataset = dataset
        self.smoothing_fwhm = smoothing_fwhm
        self.mask = mask
        self.high_pass = high_pass
        self.type_of_modeling = type_of_modeling
        self.trials_of_interest = trials_of_interest
        self.trials_to_ignore = trials_to_ignore
        self.hrf_model = hrf_model
        self.t_r = dataset_infos.t_r
        self.subjects = dataset_infos.subjects
        self.task = dataset_infos.task
        self.runs = dataset_infos.runs,
        self.conditions = dataset_infos.conditions
        self.source_dir = dataset_infos.source_dir
        self.derivatives_dir = dataset_infos.derivatives_dir
        self.out_dir = dataset_infos.out_dir
        self.niimgs = niimgs
        self.events = events
        self.confounds = confounds
        self.model = FirstLevelModel(mask=self.mask,
                                     smoothing_fwhm=self.smoothing_fwhm,
                                     high_pass=self.high_pass, t_r=self.t_r)
        self.decoder = decoder
        self.time_between_blocks = time_between_blocks
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, design_matrices=None):

        # To add : inputting list of design_matrix with check that it has the
        # right length. Not implementing for now

        # To add mumford : add an addition loop level
        '''for trial_of_interest in trials_of_interest:
            fit_design_matrix(trials_of_interest=[trial_of_interest])
                                  '''

        filenames = Parallel(n_jobs=self.n_jobs)(delayed(
            fit_design_matrix)
            (fmri, self.model, self.out_dir, None, event, confound,
             self.type_of_modeling, self.trials_of_interest,
             self.trials_to_ignore, self.hrf_model, self.time_between_blocks,
             self.verbose)
            for fmri, event, confound in zip(self.niimgs,
                                             self.events, self.confounds))

        self.glm_files = np.hstack(filenames)

        file_infos = [file.split('/')[-1].split('_')
                      for file in self.glm_files]

        self.glm_files_subs = np.asarray([file[0] for file in file_infos])
        self.glm_files_runs = np.asarray([file[1] for file in file_infos])
        self.glm_files_labels = np.asarray([file[2] for file in file_infos])

        self.decoder.fit(self.glm_files, self.glm_files_labels,
                         groups=self.glm_files_subs)


# Quick to run even with n_jobs=1, example on ds107
dataset_infos = demo_datasets("ds000107")
mask_ds107 = opj(dataset_infos.out_dir, "resampled_MNI_mask_gm.nii.gz")
pipeline = InterSubjectPipelineGLMDecoding(dataset="ds000107", smoothing_fwhm=5,
                                           mask=mask_ds107, high_pass=.01,
                                           type_of_modeling="block-design",
                                           time_between_blocks=10000,
                                           decoder=Decoder(), n_jobs=1)
pipeline.fit()
print(pipeline.decoder.cv_scores_)


# Example, quick with n_jobs=10

dataset_infos = demo_datasets("ds000117")
mask_ds117 = opj(dataset_infos.out_dir, "resampled_MNI_mask_gm.nii.gz")
pipeline = InterSubjectPipelineGLMDecoding(dataset="ds000117", smoothing_fwhm=5,
                                           mask=mask_ds117, high_pass=.01,
                                           type_of_modeling="event-related",
                                           decoder=Decoder(), n_jobs=10)
pipeline.fit()
print(pipeline.decoder.cv_scores_)

# Example on Haxby dataset quick with n_jobs=10

dataset_infos = demo_datasets("ds000105")
mask_ds105 = opj(dataset_infos.out_dir, "resampled_MNI_mask_gm.nii.gz")
pipeline = InterSubjectPipelineGLMDecoding(dataset="ds000105", smoothing_fwhm=5,
                                           mask=mask_ds105, high_pass=.01,
                                           type_of_modeling="event-related",
                                           trials_of_interest=['scissors', 'face', 'cat', 'shoe',
                                                               'house', 'scrambledpix', 'bottle', 'chair'],
                                           decoder=Decoder(), n_jobs=10)
pipeline.fit()
print(pipeline.decoder.cv_scores_)
