import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

ratings = pd.read_csv('data/sat_ratings.csv')
improvement = pd.read_csv('data/improvement.csv')
steering_llms = pd.read_csv('data/steering_llms.csv')

def rating_first_last_avgs(df):
    # Calculate mean rating and SE for attempt == 1
    mean_rating_1 = df[df['attempt'] == 1].groupby('model')['rating'].agg(['mean', 'sem']).reset_index()
    mean_rating_1.columns = ['model', 'first_attempt_avg', 'first_attempt_sem']
    
    # Calculate mean rating and SE for attempt == 5
    mean_rating_5 = df[df['attempt'] == 5].groupby('model')['rating'].agg(['mean', 'sem']).reset_index()
    mean_rating_5.columns = ['model', 'last_attempt_avg', 'last_attempt_sem']
    metrics = pd.merge(mean_rating_1, mean_rating_5, on='model')
    return metrics

def rating_avgs(df):
    avg = df.groupby('model')['rating'].agg(['mean', 'sem']).reset_index()
    avg.columns = ['model', 'rating_avg', 'rating_sem']
    return avg

def very_satisfied(df):
    res = df.groupby('model')
    # Calculate the mean of ratings >= 4 for each model
    very_satisfied_count = res['rating'].apply(lambda x: (x >= 4).sum())
    total_count = res.size()
    mean = very_satisfied_count / total_count
    sem = np.sqrt(mean * (1 - mean) / total_count)  # Adjusted SEM calculation
    res = pd.DataFrame({
        'model': very_satisfied_count.index,
        'very_satisfied_avg': mean.values,
        'very_satisfied_sem': sem.values
    })
    return res

def majority_vote(group):
    return group.mode()[0]  

def find_improvement_metrics(df):
    improvement = df.groupby(['model', 'target_headline'])

    # Randomly shuffle and group into sets of three
    grouped = improvement.apply(lambda x: x['last_chosen'].sample(frac=1).reset_index(drop=True).groupby(np.arange(len(x)) // 3).agg(majority_vote))
    # Calculate the mean and SEM over goalerence images
    mean_over_goalerence = grouped.groupby('model').agg(['mean', 'sem']).reset_index()
    mean_over_goalerence.columns = ['model', 'last_over_first_avg', 'last_over_first_sem']

    
    return mean_over_goalerence


def cohens_kappa(df, threshold, column='last_chosen'):
    values = []
    for i in range(30):
        values.append(calculate_inter_annotator_agreement_maj(df, threshold, column))
    return np.mean(values)

def calculate_inter_annotator_agreement_maj(df, threshold, column='last_chosen'):
    # Get unique user ids
    goal_imgs = df['target_headline'].unique()
    # Initialize a list to keep track of kappa scores
    y1 = []
    y2 = []

    # Iterate over all pairs of users
    for goal_img in goal_imgs:
        # Get the ratings for both users
        filtered = df[df['target_headline'] == goal_img][column]
        if len(filtered) >= threshold:
            samples = filtered.sample(n=threshold, random_state=None)
            set1 = samples.iloc[:threshold//2]
            set2 = samples.iloc[threshold//2:]
            majority1 = set1.mode().iloc[0]
            majority2 = set2.mode().iloc[0]
            y1.append(majority1)
            y2.append(majority2)

    kappa = cohen_kappa_score(y1, y2)
    
    return kappa

def overall_metrics(ratings, improvement):
    avg_rating = pd.DataFrame({
        'rating_avg': [ratings['rating'].mean()],
        'rating_sem': [ratings['rating'].sem()]
    })

    first_attempt = pd.DataFrame({
        'first_attempt_avg': [ratings[ratings['attempt'] == 1]['rating'].mean()],
        'first_attempt_sem': [ratings[ratings['attempt'] == 1]['rating'].sem()]
    })

    last_attempt = pd.DataFrame({
        'last_attempt_avg': [ratings[ratings['attempt'] == 5]['rating'].mean()],
        'last_attempt_sem': [ratings[ratings['attempt'] == 5]['rating'].sem()]
    })

    very_satisfied_avg = pd.DataFrame({
        'very_satisfied_avg': [ratings[ratings['rating'] >= 4].shape[0] / ratings.shape[0]],
        'very_satisfied_sem': [np.sqrt(ratings[ratings['rating'] >= 4].shape[0] / ratings.shape[0] * (1 - ratings[ratings['rating'] >= 4].shape[0] / ratings.shape[0]))]
    })

    improvement_grouped = improvement.groupby('target_headline').apply(lambda x: x['last_chosen'].sample(frac=1).reset_index(drop=True).groupby(np.arange(len(x)) // 3).agg(majority_vote))
    improvement_avg = pd.DataFrame({
        'last_over_first_avg': [improvement_grouped.mean()],
        'last_over_first_sem': [improvement_grouped.sem()]
    })

    # New code to merge all metrics
    combined = pd.concat([avg_rating, first_attempt, last_attempt, improvement_avg, very_satisfied_avg], axis=1)
    
    return combined


def join_metrics():
    # Calculate all metrics
    rating_metrics = rating_first_last_avgs(ratings)
    avg_metrics = rating_avgs(ratings)
    improvement_metrics = find_improvement_metrics(improvement)
    very_satisfied_metrics = very_satisfied(ratings)


    # Merge all metrics on 'model'
    combined_metrics = rating_metrics.merge(avg_metrics, on='model', how='outer') \
                                       .merge(improvement_metrics, on='model', how='outer') \
                                       .merge(very_satisfied_metrics, on='model', how='outer')
    
    # New code to add a final row with averages
    overall = overall_metrics(ratings, improvement)
    overall['model'] = 'Overall'  # Set the model name for the average row
    combined_metrics = pd.concat([combined_metrics, overall], ignore_index=True)
    
    return combined_metrics

if __name__ == '__main__':
    text_steering_llm_metrics = join_metrics()
    # text_steering_llm_metrics.to_csv('data/text_steering_llm_metrics.csv', index=False)

    print("number of steering users", steering_llms['steering_user_id'].nunique())
    print("Number of text steering headlines", steering_llms['target_headline'].nunique())
    print("number of users, ratings", ratings['user_id'].nunique(), len(ratings))
    print("Number of users, first last", improvement['user_id'].nunique(), len(improvement))