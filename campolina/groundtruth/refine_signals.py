import os
import argparse
import pod5
import pysam
import numpy as np

from tqdm import tqdm
from remora import io, refine_signal_map, util


def get_f_read_pairs(pod5_dir):
    read_reader_pair = {}
    if pod5_dir.endswith('pod5'):
        reader = pod5.Reader(f'{pod5_dir}')
        read_ids = [str(r) for r in reader.read_ids]
        read_reader_pair.update(dict.fromkeys(read_ids, reader))
        return read_reader_pair
    for f in os.listdir(pod5_dir):
        if f.endswith('pod5'):
            reader = pod5.Reader(f'{pod5_dir}/{f}')
            read_ids = [str(r) for r in reader.read_ids]
            read_reader_pair.update(dict.fromkeys(read_ids, reader))
    return read_reader_pair

def main(args):
    read_reader_pairs = get_f_read_pairs(args.pod5)
    can_bam_fh = pysam.AlignmentFile(args.bam, 'rb', check_sq=False)
    out_bam_fh = pysam.AlignmentFile(args.save_bam, 'wb', template=can_bam_fh)

    
    sig_map_refiner = refine_signal_map.SigMapRefiner(
            kmer_model_filename=args.level_table,
            scale_iters=0,
            do_fix_guage=True)

    
    for bam_read in tqdm(can_bam_fh):
        if bam_read.is_supplementary or bam_read.is_secondary or bam_read.is_unmapped:
            continue
        try:
            pod5_reader = read_reader_pairs[bam_read.query_name]
            pod5_read = next(pod5_reader.reads(selection=[bam_read.query_name]))
            io_read = io.Read.from_pod5_and_alignment(pod5_read, bam_read)
            io_read.set_refine_signal_mapping(sig_map_refiner, ref_mapping=True)
            bam_read.set_tag('RR', io_read.ref_to_signal.tolist(), replace=False)
            out_bam_fh.write(bam_read)
        except:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod5', default='47cdf38b-71d4-45c5-ba7b-bc8fcb45be31.pod5')
    parser.add_argument('--bam', default='refined_47cdf38b-71d4-45c5-ba7b-bc8fcb45be31_dorado_ref.bam')
    parser.add_argument('--level_table', default='9mer_levels_RNA004.txt')
    parser.add_argument('--save_bam', default='test.bam')
    
    args = parser.parse_args()
    main(args)
