import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

sns.set(style='darkgrid')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def cicids_preprocess(path="datasets/chethuhn/network-intrusion-dataset/versions/1"):
    data1 = pd.read_csv(f'{path}/Monday-WorkingHours.pcap_ISCX.csv')
    data1 = pd.read_csv(f'{path}/Monday-WorkingHours.pcap_ISCX.csv')
    data2 = pd.read_csv(f'{path}/Tuesday-WorkingHours.pcap_ISCX.csv')
    data3 = pd.read_csv(f'{path}/Wednesday-workingHours.pcap_ISCX.csv')
    data4 = pd.read_csv(f'{path}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
    data5 = pd.read_csv(f'{path}/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
    data6 = pd.read_csv(f'{path}/Friday-WorkingHours-Morning.pcap_ISCX.csv')
    data7 = pd.read_csv(f'{path}/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
    data8 = pd.read_csv(f'{path}/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
    # Loading the dataset
    # data1 = pd.read_csv('/home/ntenna_ech/fyp/datasets/chethuhn/network-intrusion-dataset/versions/1/Monday-WorkingHours.pcap_ISCX.csv')
    # data2 = pd.read_csv('/home/ntenna_ech/fyp/datasets/chethuhn/network-intrusion-dataset/versions/1/Tuesday-WorkingHours.pcap_ISCX.csv')
    # data3 = pd.read_csv('/home/ntenna_ech/fyp/datasets/chethuhn/network-intrusion-dataset/versions/1/Wednesday-workingHours.pcap_ISCX.csv')
    # data4 = pd.read_csv('/home/ntenna_ech/fyp/datasets/chethuhn/network-intrusion-dataset/versions/1/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
    # data5 = pd.read_csv('/home/ntenna_ech/fyp/datasets/chethuhn/network-intrusion-dataset/versions/1/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
    # data6 = pd.read_csv('/home/ntenna_ech/fyp/datasets/chethuhn/network-intrusion-dataset/versions/1/Friday-WorkingHours-Morning.pcap_ISCX.csv')
    # data7 = pd.read_csv('/home/ntenna_ech/fyp/datasets/chethuhn/network-intrusion-dataset/versions/1/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
    # data8 = pd.read_csv('/home/ntenna_ech/fyp/datasets/chethuhn/network-intrusion-dataset/versions/1/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

    data_list = [data1, data2, data3, data4, data5, data6, data7, data8]

    print('Data dimensions: ')
    data_list = [data1, data2, data3, data4, data5, data6, data7, data8]

    print('Data dimensions: ')
    for i, data in enumerate(data_list, start = 1):
        rows, cols = data.shape
        print(f'Data{i} -> {rows} rows, {cols} columns')


    
    data = pd.concat(data_list)
    rows, cols = data.shape

    print('New dimension:')
    print(f'Number of rows: {rows}')
    print(f'Number of columns: {cols}')
    print(f'Total cells: {rows * cols}')

    for d in data_list: del d

    col_names = {col: col.strip() for col in data.columns}
    data.rename(columns = col_names, inplace = True)
    data.columns
    data.info()
    
    pd.options.display.max_rows = 80

    print('Overview of Columns:')
    data.describe().transpose()
    
    pd.options.display.max_columns = 80
    data

    dups = data[data.duplicated()]
    print(f'Number of duplicates: {len(dups)}')


    data.drop_duplicates(inplace = True)
    data.shape

    #identify the missing values
    missing_val = data.isna().sum()
    print(missing_val.loc[missing_val > 0])

    numeric_cols = data.select_dtypes(include = np.number).columns
    inf_count = np.isinf(data[numeric_cols]).sum()
    print(inf_count[inf_count > 0])
    # Replacing any infinite values (positive or negative) with NaN (not a number)
    
    print(f'Initial missing values: {data.isna().sum().sum()}')
    data.replace([np.inf, -np.inf], np.nan, inplace = True)
    print(f'Missing values after processing infinite values: {data.isna().sum().sum()}')
    
    missing = data.isna().sum()
    print(missing.loc[missing > 0])

    mis_per = (missing / len(data)) * 100
    mis_table = pd.concat([missing, mis_per.round(2)], axis = 1)
    mis_table = mis_table.rename(columns = {0 : 'Missing Values', 1 : 'Percentage of Total Values'})
    print(mis_table.loc[mis_per > 0])

    sns.set_palette('pastel')
    colors = sns.color_palette()

    missing_vals = [col for col in data.columns if data[col].isna().any()]

    fig, ax = plt.subplots(figsize = (2, 6))
    msno.bar(data[missing_vals], ax = ax, fontsize = 12, color = colors)
    ax.set_xlabel('Features', fontsize = 12)
    ax.set_ylabel('Non-Null Value Count', fontsize = 12)
    ax.set_title('Missing Value Chart', fontsize = 12)
 

    med_flow_bytes = data['Flow Bytes/s'].median()
    med_flow_packets = data['Flow Packets/s'].median()

    print('Median of Flow Bytes/s: ', med_flow_bytes)
    print('Median of Flow Packets/s: ', med_flow_packets)
    
    data['Flow Bytes/s'].fillna(med_flow_bytes, inplace = True)
    data['Flow Packets/s'].fillna(med_flow_packets, inplace = True)

    print('Number of \'Flow Bytes/s\' missing values:', data['Flow Bytes/s'].isna().sum())
    print('Number of \'Flow Packets/s\' missing values:', data['Flow Packets/s'].isna().sum())

    data['Label'].unique()
    data['Label'].value_counts()

    # Creating a dictionary that maps each label to its attack type
    attack_map = {
        'BENIGN': 'BENIGN',
        'DDoS': 'DDoS',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'PortScan': 'Port Scan',
        'FTP-Patator': 'Brute Force',
        'SSH-Patator': 'Brute Force',
        'Bot': 'Bot',
        'Web Attack � Brute Force': 'Web Attack',
        'Web Attack � XSS': 'Web Attack',
        'Web Attack � Sql Injection': 'Web Attack',
        'Infiltration': 'Infiltration',
        'Heartbleed': 'Heartbleed'
    }

    # Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
    data['Attack Type'] = data['Label'].map(attack_map)

    data['Attack Type'].value_counts()

    data.drop('Label', axis = 1, inplace = True)


    le = LabelEncoder()
    data['Attack Number'] = le.fit_transform(data['Attack Type'])
    print(data['Attack Number'].unique())

    encoded_values = data['Attack Number'].unique()
    for val in sorted(encoded_values):
        print(f"{val}: {le.inverse_transform([val])[0]}")
  
    corr = data.corr(numeric_only = True).round(2)
    corr.style.background_gradient(cmap = 'coolwarm', axis = None).format(precision = 2)

    fig, ax = plt.subplots(figsize = (24, 24))
    sns.heatmap(corr, cmap = 'coolwarm', annot = False, linewidth = 0.5)
    plt.title('Correlation Matrix', fontsize = 18)
    # plt.show()

    pos_corr_features = corr['Attack Number'][(corr['Attack Number'] > 0) & (corr['Attack Number'] < 1)].index.tolist()

    print("Features with positive correlation with 'Attack Number':\n")
    for i, feature in enumerate(pos_corr_features, start = 1):
        corr_value = corr.loc[feature, 'Attack Number']
        print('{:<3} {:<24} :{}'.format(f'{i}.', feature, corr_value))

    print(f'Number of considerable important features: {len(pos_corr_features)}')

    std = data.std(numeric_only = True)
    zero_std_cols = std[std == 0].index.tolist()
    zero_std_cols

    sample_size = int(0.2 * len(data)) # 20% of the original size
    sampled_data = data.sample(n = sample_size, replace = False, random_state = 0)
    sampled_data.shape

    numeric_cols = data.select_dtypes(include = [np.number]).columns.tolist()
    print('Descriptive Statistics Comparison (mean):\n')
    print('{:<32s}{:<22s}{:<22s}{}'.format('Feature', 'Original Dataset', 'Sampled Dataset', 'Variation Percentage'))
    print('-' * 96)

    high_variations = []
    for col in numeric_cols:
            old = data[col].describe()[1]
            new = sampled_data[col].describe()[1]
            if old == 0:
                pct = 0
            else:
                pct = abs((new - old) / old)
            if pct * 100 > 5:
                high_variations.append((col, pct * 100))
            print('{:<32s}{:<22.6f}{:<22.6f}{:<2.2%}'.format(col, old, new, pct))
   
    labels = [t[0] for t in high_variations]
    values = [t[1] for t in high_variations]

    colors = sns.color_palette('Blues', n_colors=len(labels))
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.bar(labels, values, color = colors)

    for i in range(len(labels)):
        ax.text(i, values[i], str(round(values[i], 2)), ha = 'center', va = 'bottom', fontsize = 10)

    plt.xticks(rotation = 90)
    ax.set_title('Variation percenatge of the features of the sample which\n mean value variates higher than 5% of the actual mean')
    ax.set_ylabel('Percentage (%)')
    ax.set_yticks(np.arange(0, 41, 5))
    # plt.show()

    indent = '{:<3} {:<30}: {}'
    print('Unique value count for: ')
    for i, feature in enumerate(list(sampled_data.columns)[:-1], start = 1):
        print(indent.format(f'{i}.', feature, sampled_data[feature].nunique()))

    '''Generating a set of visualizations for columns that have more than one unique value but less than 50 unique values.
    For categorical columns, a bar plot is generated showing the count of each unique value.
    For numerical columns, a histogram is generated.'''
    unique_values = sampled_data.nunique()
    selected_cols = sampled_data[[col for col in sampled_data if 1 < unique_values[col] < 50]]
    rows, cols = selected_cols.shape
    col_names = list(selected_cols)
    num_of_rows = (cols + 3) // 4

    color_palette = sns.color_palette('Blues', n_colors = 3)
    plt.figure(figsize = (6 * 4, 8 * num_of_rows))

    for i in range(cols):
        plt.subplot(num_of_rows, 4, i + 1)
        col_data = selected_cols.iloc[:, i]
        if col_data.dtype.name == 'object':
            col_data.value_counts().plot(kind = 'bar', color = color_palette[2])
        else:
            col_data.hist(color = color_palette[0])

        plt.ylabel('Count')
        plt.xticks(rotation = 90)
        plt.title(col_names[i])

    plt.tight_layout()
    # plt.show()

    # Correlation matrix for sampled data
    corr_matrix = sampled_data.corr(numeric_only = True).round(2)
    corr_matrix.style.background_gradient(cmap = 'coolwarm', axis = None).format(precision = 2)

    # Plotting the pairs of strongly positive correlated features in the sampled_data that have a correlation coefficient of 0.85 or higher
    cols = list(sampled_data.columns)[:-2]
    high_corr_pairs = []
    corr_th = 0.85

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = sampled_data[cols[i]].corr(sampled_data[cols[j]])
        # If the correlation coefficient is NaN or below the threshold, skip to the next pair
            if np.isnan(val) or val < corr_th:
                continue
            high_corr_pairs.append((val, cols[i], cols[j]))

    size, cols = len(high_corr_pairs), 4
    rows, rem =  size // cols, size % cols
    if rem:
        rows += 1

    fig, axs = plt.subplots(rows, cols, figsize = (24, int(size * 1.7)))
    for i in range(rows):
        for j in range(cols):
            try:
             val, x, y = high_corr_pairs[i * cols + j]
             if val > 0.99:
                axs[i, j].scatter(sampled_data[x], sampled_data[y], color = 'green', alpha = 0.1)
             else:
                axs[i, j].scatter(sampled_data[x], sampled_data[y], color = 'blue', alpha = 0.1)
             axs[i, j].set_xlabel(x)
             axs[i, j].set_ylabel(y)
             axs[i, j].set_title(f'{x} vs\n{y} ({val:.2f})')
            except IndexError:
             fig.delaxes(axs[i, j])

    fig.tight_layout()
    # plt.show()

    sampled_data.drop('Attack Number', axis = 1, inplace = True)
    data.drop('Attack Number', axis = 1, inplace = True)

    # Identifying outliers
    numeric_data = sampled_data.select_dtypes(include = ['float', 'int'])
    q1 = numeric_data.quantile(0.25)
    q3 = numeric_data.quantile(0.75)
    iqr = q3 - q1
    outlier = (numeric_data < (q1 - 1.5 * iqr)) | (numeric_data > (q3 + 1.5 * iqr))
    outlier_count = outlier.sum()
    outlier_percentage = round(outlier.mean() * 100, 2)
    outlier_stats = pd.concat([outlier_count, outlier_percentage], axis = 1)
    outlier_stats.columns = ['Outlier Count', 'Outlier Percentage']

    print(outlier_stats)

    outlier_counts = {}
    for i in numeric_data:
        for attack_type in sampled_data['Attack Type'].unique():
            attack_data = sampled_data[i][sampled_data['Attack Type'] == attack_type]
            q1, q3 = np.percentile(attack_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            num_outliers = ((attack_data < lower_bound) | (attack_data > upper_bound)).sum()
            outlier_percent = num_outliers / len(attack_data) * 100
            outlier_counts[(i, attack_type)] = (num_outliers, outlier_percent)

    for i in numeric_data:
        print(f'Feature: {i}')
        for attack_type in sampled_data['Attack Type'].unique():
            num_outliers, outlier_percent = outlier_counts[(i, attack_type)]
            print(f'- {attack_type}: {num_outliers} ({outlier_percent:.2f}%)')
        print()

    # Plotting the percentage of outliers that are higher than 20%
    fig, ax = plt.subplots(figsize = (24, 10))
    for i in numeric_data:
        for attack_type in sampled_data['Attack Type'].unique():
            num_outliers, outlier_percent = outlier_counts[(i, attack_type)]
            if outlier_percent > 20:
                ax.bar(f'{i} - {attack_type}', outlier_percent)

    ax.set_xlabel('Feature-Attack Type')
    ax.set_ylabel('Percentage of Outliers')
    ax.set_title('Outlier Analysis')
    ax.set_yticks(np.arange(0, 41, 10))
    plt.xticks(rotation = 90)
    # plt.show()

    # Different 'Attack Type' in the main dataset excluding 'BENIGN'
    attacks = data.loc[data['Attack Type'] != 'BENIGN']

    plt.figure(figsize = (10, 6))
    ax = sns.countplot(x = 'Attack Type', data = attacks, palette = 'pastel', order = attacks['Attack Type'].value_counts().index)
    plt.title('Types of attacks')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation = 90)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height() + 1000), ha = 'center')

    # plt.show()

    attack_counts = attacks['Attack Type'].value_counts()
    threshold = 0.005
    percentages = attack_counts / attack_counts.sum()
    small_slices = percentages[percentages < threshold].index.tolist()
    attack_counts['Other'] = attack_counts[small_slices].sum()
    attack_counts.drop(small_slices, inplace = True)

    sns.set_palette('pastel')
    plt.figure(figsize = (8, 8))
    plt.pie(attack_counts.values, labels = attack_counts.index, autopct = '%1.1f%%', textprops={'fontsize': 6})
    plt.title('Distribution of Attack Types')
    plt.legend(attack_counts.index, loc = 'best')
    # plt.show()

    # Creating a boxplot for each attack type with the columns of sampled dataset
    for attack_type in sampled_data['Attack Type'].unique():
        attack_data = sampled_data[sampled_data['Attack Type'] == attack_type]
        plt.figure(figsize=(20, 20))
        sns.boxplot(data = attack_data.drop(columns = ['Attack Type']), orient = 'h')
        plt.title(f'Boxplot of Features for Attack Type: {attack_type}')
        plt.xlabel('Feature Value')
        # plt.show()

    data.groupby('Attack Type').first()

    old_memory_usage = data.memory_usage().sum() / 1024 ** 2
    print(f'Initial memory usage: {old_memory_usage:.2f} MB')
    for col in data.columns:
        col_type = data[col].dtype
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            # Downcasting float64 to float32
            if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                data[col] = data[col].astype(np.float32)

            # Downcasting int64 to int32
            elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                data[col] = data[col].astype(np.int32)

    new_memory_usage = data.memory_usage().sum() / 1024 ** 2
    print(f"Final memory usage: {new_memory_usage:.2f} MB")

    print(f'Reduced memory usage: {1 - (new_memory_usage / old_memory_usage):.2%}')
    
    data.info()
    
    data.describe().transpose()

    num_unique = data.nunique()
    one_variable = num_unique[num_unique == 1]
    not_one_variable = num_unique[num_unique > 1].index

    dropped_cols = one_variable.index
    data = data[not_one_variable]
    print("Dropped Cols" ,dropped_cols)

    features = data.drop('Attack Type', axis = 1)
    attacks = data['Attack Type']

    print(features.shape)
    print(attacks.shape)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
   

    # # size = len(features.columns) // 2
    # # ipca = IncrementalPCA(n_components = size, batch_size = 500)
    # # for batch in np.array_split(scaled_features, len(features) // 500):
    # #     ipca.partial_fit(batch)

    size = len(features.columns) // 2
    ipca = IncrementalPCA(n_components = size, batch_size = 500)
    for batch in np.array_split(scaled_features, len(features) // 500):
        ipca.partial_fit(batch)

    print(f'information retained: {sum(ipca.explained_variance_ratio_):.2%}')

    transformed_features = ipca.transform(scaled_features)
    new_data = pd.DataFrame(transformed_features, columns = [f'PC{i+1}' for i in range(size)])
    new_data['Attack Type'] = attacks.values

    normal_traffic = new_data.loc[new_data['Attack Type'] == 'BENIGN']
    intrusions = new_data.loc[new_data['Attack Type'] != 'BENIGN']

    normal_traffic = normal_traffic.sample(n = len(intrusions), replace = False)

    ids_data = pd.concat([intrusions, normal_traffic])
    ids_data['Attack Type'] = np.where((ids_data['Attack Type'] == 'BENIGN'), 0, 1)
    bc_data = ids_data.sample(n = 15000)

    print(bc_data['Attack Type'].value_counts())




    X_bc = bc_data.drop('Attack Type', axis = 1)
    y_bc = bc_data['Attack Type']

    X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size = 0.25, random_state = 0)


    new_data['Attack Type'].value_counts()
    class_counts = new_data['Attack Type'].value_counts()
    selected_classes = class_counts[class_counts > 1950]
    class_names = selected_classes.index
    selected = new_data[new_data['Attack Type'].isin(class_names)]

    dfs = []
    for name in class_names:
        df = selected[selected['Attack Type'] == name]
        if len(df) > 2500:
            df = df.sample(n = 5000, random_state = 0)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index = True)
    df['Attack Type'].value_counts()

    X = df.drop('Attack Type', axis=1)
    y = df['Attack Type']

    smote = SMOTE(sampling_strategy='auto', random_state=0)
    X_upsampled, y_upsampled = smote.fit_resample(X, y)

    blnc_data = pd.DataFrame(X_upsampled)
    blnc_data['Attack Type'] = y_upsampled
    blnc_data = blnc_data.sample(frac=1)

    blnc_data['Attack Type'].value_counts()

    features = blnc_data.drop('Attack Type', axis = 1)
    labels = blnc_data['Attack Type']
    return features,labels
    # return X_train_bc, X_test_bc, y_train_bc, y_test_bc,features,labels


    
 