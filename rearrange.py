import os
from subprocess import call
import pickle


def main(root_dir, task):
    variations = os.listdir(f'{root_dir}/{task}/all_variations/episodes')
    seen_variations = {}
    for variation in variations:
        num = int(variation.replace('episode', ''))
        variation = pickle.load(
            open(
                f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_number.pkl',
                'rb'
            )
        )
        os.makedirs(f'{root_dir}/{task}/variation{variation}/episodes', exist_ok=True)

        if variation not in seen_variations.keys():
            seen_variations[variation] = [num]
        else:
            seen_variations[variation].append(num)

        ep_id = len(seen_variations[variation]) - 1
        os.makedirs(f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}', exist_ok=True)
        call(['cp',
              "{:s}/{:s}/all_variations/episodes/episode{:d}/low_dim_obs.pkl".format(root_dir, task, num),
              f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}'])


if __name__ == '__main__':
    root_dir = '/data/group_data/katefgroup/VLA/peract2_raw_squash/train/'
    tasks = [f for f in os.listdir(root_dir) if '.zip' not in f]
    for task in tasks:
        main(root_dir, task)
