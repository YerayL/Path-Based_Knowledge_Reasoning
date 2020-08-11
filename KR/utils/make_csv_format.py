import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_data', type=str, default='./data/data_output')
parser.add_argument('-d', '--output_data', type=str, default='./data/data_output_csv')
args = parser.parse_args()

def make_one_csv(in_path, out_path,query):
    # path_num = in_path[:-4].split('.')[-1]   # Get the number of paths, if need.
    # dirname = os.path.dirname(in_path)
    with open(in_path, 'r') as f1:
        for line in f1:
            label, path = line.split('\t')
            label = label.strip()
            path = path.strip()
            entity_types = ''
            relation = ''
            path_list = path.split(';')
            for i in path_list:
                ent_rel = i.split(' ')
                for j in ent_rel:
                    entity_types += ','.join(j.split(',')[:7]) + ' '
                    relation += j.split(',')[-1] + ' '
                entity_types = entity_types.strip() + ';'
                relation = relation.strip() + ';'
            entity_types = entity_types[:-1]
            relation = relation[:-1]
            new_line = label + '\t' + entity_types + '\t' + relation + '\t' + str(query) + '\n'
            # print(new_line)
            with open(out_path, 'a') as f2:
                f2.write(new_line)



# input_dir_list = os.listdir(input_data)
# args.input_data = os.path.join(os.getcwd(), 'data\\data_output')  # if linux, please annotate this line

counter = 0
dir_list = os.listdir(args.input_data)
for i in dir_list:
    print(f'index:{counter},relation:{i}')
    input_data = out_path = os.path.join(args.input_data, i)
    for root, dirs, files in os.walk(input_data, topdown=True):
        print(root)
        for file in files:
            if file.endswith('.int'):
                path_num = file[:-4].split('.')[-1]
                # data = root.split('\\')[-1] + '.' + path_num + '.csv'   # if windows
                data = root.split('/')[-1] + '.' + path_num + '.csv'  # elif linux
                new_root = os.path.join(args.output_data, '/'.join(root.split('/')[-2:]))
                out_path = os.path.join(new_root, data)
                # print(f'out path is \'{out_path}\'.')
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                make_one_csv(os.path.join(root, file), out_path, counter)
    counter += 1





