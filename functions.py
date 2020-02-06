import pandas as pd
from string import ascii_letters, digits
from surprise.model_selection import cross_validate

### DataFrame operations

def k_from_details(details):
    try:
        return details['actual_k']
    except KeyError:
        return 1000

def short_title(title, max_len=40):
    title = str(title).split(' ')
    short_title = ''

    for i in range(len(title)):
        if len(short_title) < max_len:
            short_title = ' '.join([short_title, title[i]])
    short_title = short_title.strip()
    return short_title

def ascii_check(item):
    for letter in str(item):
        if letter not in ascii_letters + digits:
            return 1
        else:
            return 0

def ascii_check_bulk(df):
    for col in df.columns:
        print('items with non-ascii characters in %s: %d' % (col, df[col].apply(ascii_check).sum()))
    print('')

def colname_fix(colname):
    return colname.lower().replace('-','_')

### Model-related functions

def get_model_name(model):
    return str(model).split('.')[-1].split(' ')[0].replace("'>", "")

def cv_multiple_models(data, models_dict):
    results = pd.DataFrame()

    for model_name, model in models_dict.items():
        print('\n---> CV for %s...' % model_name)

        cv_results = cross_validate(model, data)
        tmp = pd.DataFrame(cv_results).mean()
        tmp['model'] = model_name
        results = results.append(tmp, ignore_index=True)

    return results

def generate_models_dict(models, sim_names, user_based):
    models_dict = {}

    for sim_name in sim_names:
        sim_dict = {
            'name': sim_name,
            'user_based': user_based
        }
        for model in models:
            model_name = get_model_name(model) + ' ' + sim_name
            models_dict[model_name] = model(sim_options=sim_dict)

    return models_dict

def draw_model_results(results):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.xticks(rotation=90)

    palette = sns.color_palette("RdBu", len(results))

    sns.barplot(x='model', y='test_rmse', data=results, palette=palette, ax=ax1)
    ax1.set_title('Test RMSE and fit time of evaluated models')

    ax2 = ax1.twinx()
    sns.scatterplot(x='model', y='fit_time', data=results, color='black', ax=ax2)
    ax2.set(ylim=(0, results['fit_time'].max() * 1.1))

    plt.show()