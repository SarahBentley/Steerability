"""
This script calculates useful metrics for measuring the steerability of image generation models.
It processes data from multiple CSV files containing human and automated ratings to generate summary statistics and inter-annotator agreement measures.

The script handles several types of data:
- Ratings of the similarity of generated imges to goal images on 4-point and 10-point scales
- Improvement assessments between users' first and last image generations
- Prompt-Output-Misalignment (POM) data
- CLIP and DreamSim similarity scores

Input files expected in data/:
- pom1.csv: Prompt-Output-Misalignment (POM) data for first attempts
- pom5.csv: POM data for fifth attempts
- improvement.csv: Improvement assessments between users' first and last image generations
- sat_rating_10.csv: Satisfaction ratings on 10-point scale
- sat_rating_4.csv: Satisfaction ratings on 4-point scale
- steering.csv: CLIP and DreamSim similarity scores on steering data from all users in our study.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Load input data files
pom1 = pd.read_csv('data/pom1.csv')
pom5 = pd.read_csv('data/pom5.csv')
improvement = pd.read_csv('data/improvement.csv')
ratings10 = pd.read_csv('data/sat_rating_10.csv')
ratings4 = pd.read_csv('data/sat_rating_4.csv')
steering = pd.read_csv('data/steering.csv')

def majority_vote(group):
    """
    Helper function to calculate the mode (majority vote) of a group of values.
    
    Args:
        group: Series or array-like object
        
    Returns:
        Most common value in the group
    """
    return group.mode()[0]  

def rating_improvement_avgs(df):
    """
    Calculate average ratings and standard errors for first and last attempts.
    
    Args:
        df (pd.DataFrame): DataFrame containing rating data with 'attempt' and 'rating' columns
        
    Returns:
        pd.DataFrame: DataFrame with average ratings and standard errors for first and last attempts
    """
    # Calculate mean rating and SE for attempt == 1
    mean_rating_1 = df[df['attempt'] == 1].groupby('model')['rating'].agg(['mean', 'sem']).reset_index()
    mean_rating_1.columns = ['model', 'first_attempt_avg', 'first_attempt_sem']
    
    # Calculate mean rating and SE for attempt == 5
    mean_rating_5 = df[df['attempt'] == 5].groupby('model')['rating'].agg(['mean', 'sem']).reset_index()
    mean_rating_5.columns = ['model', 'last_attempt_avg', 'last_attempt_sem']
    metrics = pd.merge(mean_rating_1, mean_rating_5, on='model')
    return metrics

def rating_avgs(df):
    """
    Calculate overall rating averages and maximum ratings per goal image.
    
    Args:
        df (pd.DataFrame): DataFrame containing rating data
        
    Returns:
        pd.DataFrame: DataFrame with average ratings, standard errors, and maximum ratings
    """
    avg = df.groupby('model')['rating'].agg(['mean', 'sem']).reset_index()
    avg.columns = ['model', 'rating_avg', 'rating_sem']
    maxes = (df.groupby(['model', 'goal_image'])['rating']
             .max()  # first get max rating for each goal image within each model
             .groupby('model')  # then group just by model
             .agg(['mean', 'sem'])  # calculate mean and sem of those maxes
             .reset_index())
    maxes.columns = ['model', 'rating_max_avg', 'rating_max_sem']
    all = pd.merge(avg, maxes, on='model')
    return all

def other_avgs(df, column):
    """
    Calculate averages and standard errors for a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Name of column to analyze
        
    Returns:
        pd.DataFrame: DataFrame with averages and standard errors for specified column
    """
    avg = df.groupby('model')[column].agg(['mean', 'sem']).reset_index()
    avg.columns = ['model', column+'_avg', column+'_sem']
    return avg

def other_improvement_avgs(df, column):
    """
    Calculate improvement metrics for a specified column between first and last attempts.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Name of column to analyze
        
    Returns:
        pd.DataFrame: DataFrame with first and last attempt metrics
    """
    # Calculate mean rating and SE for attempt == 1
    mean_rating_1 = df[df['attempt'] == 1].groupby('model')[column].agg(['mean', 'sem']).reset_index()
    mean_rating_1.columns = ['model', column+'_first_attempt_avg', column+'_first_attempt_sem']
    
    # Calculate mean rating and SE for attempt == 5
    mean_rating_5 = df[df['attempt'] == 5].groupby('model')[column].agg(['mean', 'sem']).reset_index()
    mean_rating_5.columns = ['model', column+'_last_attempt_avg', column+'_last_attempt_sem']
    metrics = pd.merge(mean_rating_1, mean_rating_5, on='model')
    return metrics

def find_improvement_metrics(df):
    """
    Calculate improvement metrics using majority voting over grouped samples.
    """
    # Group by model and goal_image, but keep model column
    grouped = (df.groupby(['model', 'goal_image'])['last_chosen']
              .apply(lambda x: x.sample(frac=1).reset_index(drop=True)
                    .groupby(np.arange(len(x)) // 3)
                    .agg(majority_vote))
              .reset_index(level='goal_image', drop=True))  # Only drop goal_image index
    
    # Calculate the mean and SEM over goal images
    mean_over_goal = grouped.groupby('model').agg(['mean', 'sem']).reset_index()
    mean_over_goal.columns = ['model', 'last_over_first_avg', 'last_over_first_sem']
    
    return mean_over_goal

def find_pom1_metrics(df):
    """
    Calculate Prompt-Output-Misalignment for first attempts using majority voting.
    """
    # Group by model and goal_image, but keep model column
    grouped = (df.groupby(['model', 'goal_image'])['ref_chosen']
              .apply(lambda x: x.sample(frac=1).reset_index(drop=True)
                    .groupby(np.arange(len(x)) // 3)
                    .agg(majority_vote))
              .reset_index(level='goal_image', drop=True))  # Only drop goal_image index
    
    # Calculate the mean and SEM over goal images
    mean_over_goal = grouped.groupby('model').agg(['mean', 'sem']).reset_index()
    mean_over_goal.columns = ['model', 'pom1_avg', 'pom1_sem']
    
    return mean_over_goal

def find_pom5_metrics(df):
    """
    Calculate Prompt-Output-Misalignment for fifth attempts using majority voting.
    """
    # Group by model and goal_image, but keep model column
    grouped = (df.groupby(['model', 'goal_image'])['ref_chosen']
              .apply(lambda x: x.sample(frac=1).reset_index(drop=True)
                    .groupby(np.arange(len(x)) // 3)
                    .agg(majority_vote))
              .reset_index(level='goal_image', drop=True))  # Only drop goal_image index
    
    # Calculate the mean and SEM over goal images
    mean_over_goal = grouped.groupby('model').agg(['mean', 'sem']).reset_index()
    mean_over_goal.columns = ['model', 'pom5_avg', 'pom5_sem']
    
    return mean_over_goal


def cohens_kappa(df, threshold, column='last_chosen'):
    """
    Calculate Cohen's Kappa score for inter-annotator agreement.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (int): Minimum number of samples required
        column (str): Column name to analyze (default: 'last_chosen')
        
    Returns:
        float: Average Cohen's Kappa score over 30 iterations
    """
    values = []
    for i in range(30):
        values.append(calculate_inter_annotator_agreement_maj(df, threshold, column))
    return np.mean(values)

def calculate_inter_annotator_agreement_maj(df, threshold, column='last_chosen'):
    """
    Calculate inter-annotator agreement using majority voting.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (int): Minimum number of samples required for majority voting
        column (str): Column name to analyze (default: 'last_chosen')
        
    Returns:
        float: Cohen's Kappa score for inter-annotator agreement
    """
    # Get unique user ids
    ref_imgs = df['goal_image'].unique()
    # Initialize a list to keep track of kappa scores
    y1 = []
    y2 = []

    # Iterate over all pairs of users
    for ref_img in ref_imgs:
        # Get the ratings for both users
        filtered = df[df['goal_image'] == ref_img][column]
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

def overall_metrics(ratings, improvement, pom1, pom5, steering):
    """
    Calculate overall metrics across all experiments and models.
    
    Args:
        ratings (pd.DataFrame): Ratings data
        improvement (pd.DataFrame): Improvement data
        pom1 (pd.DataFrame): Prompt-Output-Misalignment first attempt data
        pom5 (pd.DataFrame): Prompt-Output-Misalignment fifth attempt data
        steering (pd.DataFrame): Steering metrics data (DreamSim and CLIP similarity scores)
        
    Returns:
        pd.DataFrame: Combined metrics with averages and standard errors
    """
    avg_rating = pd.DataFrame({
        'rating_avg': [ratings['rating'].mean()],
        'rating_sem': [ratings['rating'].sem()]
    })

    max_rating = pd.DataFrame({
        'rating_max_avg': [ratings.groupby(['goal_image'])['rating'].max().mean()],
        'rating_max_sem': [ratings.groupby(['goal_image'])['rating'].max().sem()]
    })

    dreamsim_avg = pd.DataFrame({
        'dreamsim_avg': [steering['dreamsim'].mean()],
        'dreamsim_sem': [steering['dreamsim'].sem()]
    })

    clip_avg = pd.DataFrame({
        'clip_similarity_avg': [steering['clip_similarity'].mean()],
        'clip_similarity_sem': [steering['clip_similarity'].sem()]
    })

    first_attempt = pd.DataFrame({
        'first_attempt_avg': [ratings[ratings['attempt'] == 1]['rating'].mean()],
        'first_attempt_sem': [ratings[ratings['attempt'] == 1]['rating'].sem()]
    })

    dreamsim_first_attempt = pd.DataFrame({
        'dreamsim_first_attempt_avg': [steering[steering['attempt'] == 1]['dreamsim'].mean()],
        'dreamsim_first_attempt_sem': [steering[steering['attempt'] == 1]['dreamsim'].sem()]
    })

    clip_first_attempt = pd.DataFrame({
        'clip_similarity_first_attempt_avg': [steering[steering['attempt'] == 1]['clip_similarity'].mean()],
        'clip_similarity_first_attempt_sem': [steering[steering['attempt'] == 1]['clip_similarity'].sem()]
    })

    last_attempt = pd.DataFrame({
        'last_attempt_avg': [ratings[ratings['attempt'] == 5]['rating'].mean()],
        'last_attempt_sem': [ratings[ratings['attempt'] == 5]['rating'].sem()]
    })

    dreamsim_last_attempt = pd.DataFrame({
        'dreamsim_last_attempt_avg': [steering[steering['attempt'] == 5]['dreamsim'].mean()],
        'dreamsim_last_attempt_sem': [steering[steering['attempt'] == 5]['dreamsim'].sem()]
    })

    clip_last_attempt = pd.DataFrame({
        'clip_similarity_last_attempt_avg': [steering[steering['attempt'] == 5]['clip_similarity'].mean()],
        'clip_similarity_last_attempt_sem': [steering[steering['attempt'] == 5]['clip_similarity'].sem()]
    })

    improvement_grouped = (improvement.groupby('goal_image')['last_chosen']
                         .apply(lambda x: x.sample(frac=1).reset_index(drop=True)
                               .groupby(np.arange(len(x)) // 3)
                               .agg(majority_vote))
                         .reset_index(level=0, drop=True))
    improvement_avg = pd.DataFrame({
        'last_over_first_avg': [improvement_grouped.mean()],
        'last_over_first_sem': [improvement_grouped.sem()]
    })

    pom1_grouped = (pom1.groupby('goal_image')['ref_chosen']
                    .apply(lambda x: x.sample(frac=1).reset_index(drop=True)
                          .groupby(np.arange(len(x)) // 3)
                          .agg(majority_vote))
                    .reset_index(level=0, drop=True))
    pom1_avg = pd.DataFrame({
        'pom1_avg': [pom1_grouped.mean()],
        'pom1_sem': [pom1_grouped.sem()]
    })

    pom5_grouped = (pom5.groupby('goal_image')['ref_chosen']
                    .apply(lambda x: x.sample(frac=1).reset_index(drop=True)
                          .groupby(np.arange(len(x)) // 3)
                          .agg(majority_vote))
                    .reset_index(level=0, drop=True))
    pom5_avg = pd.DataFrame({
        'pom5_avg': [pom5_grouped.mean()],
        'pom5_sem': [pom5_grouped.sem()]
    })
    # New code to merge all metrics
    combined = pd.concat([avg_rating, first_attempt, last_attempt, improvement_avg, pom1_avg, pom5_avg, dreamsim_avg, clip_avg, dreamsim_first_attempt, clip_first_attempt, dreamsim_last_attempt, clip_last_attempt, max_rating], axis=1)
    
    return combined

def join_metrics():
    """
    Join all metrics into a single DataFrame and add overall averages.
    
    Returns:
        pd.DataFrame: Combined metrics for all models and an overall average row
    """
    # Calculate all metrics
    rating_metrics = rating_improvement_avgs(ratings10)
    avg_metrics = rating_avgs(ratings10)
    improvement_metrics = find_improvement_metrics(improvement)
    pom1_metrics = find_pom1_metrics(pom1)
    pom5_metrics = find_pom5_metrics(pom5)
    dreamsim_metrics = other_avgs(steering, 'dreamsim')
    clip_metrics = other_avgs(steering, 'clip_similarity')
    dreamsim_improvement_metrics = other_improvement_avgs(steering, 'dreamsim')
    clip_improvement_metrics = other_improvement_avgs(steering, 'clip_similarity')

    # Merge all metrics on 'model'
    combined_metrics = rating_metrics.merge(avg_metrics, on='model', how='outer') \
                                       .merge(improvement_metrics, on='model', how='outer') \
                                       .merge(pom1_metrics, on='model', how='outer') \
                                       .merge(pom5_metrics, on='model', how='outer') \
                                       .merge(dreamsim_metrics, on='model', how='outer') \
                                       .merge(clip_metrics, on='model', how='outer') \
                                       .merge(dreamsim_improvement_metrics, on='model', how='outer') \
                                       .merge(clip_improvement_metrics, on='model', how='outer')
    
    # New code to add a final row with averages
    overall = overall_metrics(ratings10, improvement, pom1, pom5, steering)
    overall['model'] = 'Overall'  # Set the model name for the average row
    combined_metrics = pd.concat([combined_metrics, overall], ignore_index=True)
    
    return combined_metrics

if __name__ == '__main__':
    # Calculate and display summary metrics
    summary_metrics = join_metrics()
    
    # Calculate and print various statistics
    print("Cohen's Kappa for improvement: ", cohens_kappa(improvement, 6, column='last_chosen'))
    
    # Calculate satisfaction percentages for different rating scales
    print("Percent very satisfied (10 pt scale)", ratings10[ratings10['rating'] >= 8].shape[0] / ratings10.shape[0])
    print("Percent at least somewhat satisfied (4 pt scale)", ratings4[ratings4['rating'] >= 3].shape[0] / ratings4.shape[0])
    print("Percent very satisfied (4 pt scale)", ratings4[ratings4['rating'] >= 4].shape[0] / ratings4.shape[0])
    print("Percent very unsatisfied (4 pt scale)", ratings4[ratings4['rating'] <= 1].shape[0] / ratings4.shape[0])
    print("Percent unsatisfied (4 pt scale)", ratings4[ratings4['rating'] <= 2].shape[0] / ratings4.shape[0])

