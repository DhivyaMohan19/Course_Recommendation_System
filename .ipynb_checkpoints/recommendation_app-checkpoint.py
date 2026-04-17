import streamlit as st
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import io

# Import any other libraries you need for your recommendation algorithm

# Set page title and configuration
st.set_page_config(page_title="Course Recommendation System", layout="wide")
st.title("Course Recommendation System")

# Load your data
@st.cache_data  # This caches your data loading function
def load_data():
    # Replace with your actual data loading code
    df = pd.read_csv("online_course_recommendation.csv")
    return df
   
df_encoded = load_data()

# Sidebar for user input
st.sidebar.header("User Settings")

target_user_id = -1
course_name = None
user_type = st.sidebar.selectbox("Select User Type", index= None,options = ['New User','Existing User'])

if user_type == 'Existing User':
    target_user_id = st.sidebar.selectbox("Select User ID",index= None,
                                     options = sorted(df_encoded['user_id'].unique()))  
    
    with open('kmeans_with_data.pkl', 'rb') as f:
        loaded = pickle.load(f)

    kmeans_model = loaded['model']
    test_users_labelled = loaded['data']
    count_enrollments_df = loaded['data1']
    popular_courses_by_cluster = loaded['meta']
    
elif user_type == 'New User':
    course_name = st.sidebar.selectbox("Select a Course You Prefer",index= None,
                                     options = sorted(df_encoded['course_name'].unique()))

    course_dict = pickle.load(open('course_dict.pkl', 'rb'))
    course_dict = pd.DataFrame(course_dict)
    similarity = pickle.load(open('recommendation_similarity.pkl', 'rb'))
    
else:
    st.write(f"Select Your  User Type To Get Recommendations!!!")

  
# Function to generate recommendations for a new user
def generate_content_recommendations(course,data):
    course_ids = []
    
    #st.write(course) 
    
    course_index = course_dict[course_dict['course_name'] == course].index[0]
        
    distances = similarity[course_index]
    
    #st.write(course_dict)
    #st.write(distances)
    course_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]  # skip itself    
    #course_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6] #excluding the input course 

    
    for i in course_list:
        
        if i[1] > 0:
            #st.write(data.iloc[i[0]].course_name)
            #Level --> {course_dict.iloc[i[0]].difficulty_level}: Average Course Rating --> {course_dict.iloc[i[0]].rating}")
            course_ids.append(data.iloc[i[0]].course_id)
            
    return course_ids

def generate_user_recommendations(user_id):
    user_course_ids = []
    recommended_list = []
    prev_cluster = []

    for i in range(kmeans_model.n_clusters):
        tmp_df = count_enrollments_df[count_enrollments_df['cluster']==i]
        tmp_df.sort_values(by=['enrollments'], ascending=False, inplace=True)
        popular_courses_by_cluster[i] = tmp_df['course_id'].values[:5]

    
    for index, row in test_users_labelled[test_users_labelled['user_id'] == user_id].iterrows():
        user_id = row['user_id']
        cluster = row['cluster']
        if cluster not in prev_cluster:
          # Convert the NumPy array to a Python list using .tolist()
          courses_to_add = popular_courses_by_cluster[cluster][:5].tolist()
          recommended_list.append(courses_to_add)
          #print(f'user {user_id} in cluster {cluster} is recommened courses as {courses_to_add}')
          prev_cluster.append(cluster)
            

    final_recommended_list = list(set([course for sublist in recommended_list for course in sublist]))
    
    #st.write(final_recommended_list)
    
    #print("Flattened and unique recommended courses:")
    #st.write(final_recommended_list)
    recommend_counter = 0
    course_names = []
    user_course_ids = []
    for course_id in final_recommended_list:
                
          if ((df_encoded['course_id'] == course_id) & (df_encoded['user_id'] == target_user_id)).any():
              continue
          elif recommend_counter <= 5:
              recommend_counter += 1
              course_info = df_encoded[df_encoded['course_id'] == course_id][['course_name', 'difficulty_level']].iloc[0]
              course_name = course_info['course_name']
              difficulty_level = course_info['difficulty_level'] 
              if course_name not in course_names:
                  user_course_ids.append(course_id)
                  #print(f"Course ID: {course_id}, Course Name: {course_name}, Difficulty Level: {difficulty_level}")
                  course_names.append(course_name)
              else:
                  continue
          else:
              break
    return user_course_ids
    
def recommend_courses(df, top_n=5, w_rating=0.6, w_enroll=0.4):
    
    data = df.copy()
    
    # --- Normalize enrollments (0–1) ---
    data['enroll_norm'] = (
        (data['enrollment_numbers'] - data['enrollment_numbers'].min()) /
        (data['enrollment_numbers'].max() - data['enrollment_numbers'].min())
    )
    
    # --- Normalize rating (assuming out of 5) ---
    data['rating_norm'] = data['rating'] / 5.0
    
    # --- Hybrid score ---
    data['score'] = (
        w_rating * data['rating_norm'] +
        w_enroll * data['enroll_norm']
    )
    
    # --- Sort and return top N ---
    recommendations = data.sort_values(by='score', ascending=False)

    final_result = recommendations[recommendations['score'] > 0.8]


    #st.write(recommendations)
    #return list(recommendations['course_id', 'course_name'].head(top_n))   
    #return recommendations.iloc[:top_n, recommendations.columns.get_loc('course_id')].tolist()

    #st.write(recommendations[['course_id', 'course_name']].head(top_n).values.tolist())
    
    #return final_result.head(top_n)
    #return recommendations.iloc[:top_n, recommendations.columns.get_loc('course_id')].tolist()
    return recommendations.head(top_n).index.tolist()
    
# Generate recommendations when user selects an ID
if st.sidebar.button("Generate Course Recommendations"):
    
    #st.write(target_user_id,user_type)
    
    #if user_type == 'New User'  and course_name is None:
        
            #st.write(f"Select A Preferred Course To Get Course Recommendations!!!")
        
    if user_type == 'Existing User' and target_user_id is None: 

        st.write(f"Select Your User ID To Get Course Recommendations!!!")
        
    else:
        with st.spinner("Generating recommendations..."):
            if user_type == 'Existing User':
                recommended_list = generate_user_recommendations(target_user_id)
                #st.write(recommended_list)
            else:
                if course_name is None:
                    recommended_list = recommend_courses(df_encoded)
                    #st.write(recommended_list)
                else:
                    recommended_list = generate_content_recommendations(course_name,df_encoded)
                    #st.write(recommended_list)

        
            st.subheader("Recommended Courses")
        
            # Create a container for recommendations
            recommendation_container = st.container()

            # Hybrid Recommendation System for new users
            if course_name is None and user_type == 'New User':
                final_recommended_list = df_encoded.iloc[recommended_list][['course_id', 'course_name', 'difficulty_level', 'rating']]
                #st.write(final_recommended_list)
                
                with recommendation_container:
                    for _, row in final_recommended_list.iterrows():
                        course_id = row['course_id']
                        course_name = row['course_name']
                        difficulty_level = row['difficulty_level']
                        rating = row['rating']

                        # Display course information in a nice format
                        col1, col2, col3, col4  = st.columns([1, 2, 1, 1])
                        with col1:
                            st.write(f"**Course ID:** {course_id}")
                        with col2:
                            st.write(f"**Course Name:** {course_name}")
                        with col3:
                            st.write(f"**Difficulty:** {difficulty_level}")
                        with col4:
                            st.write(f"**User Rating:** {rating}")
                        st.divider()
            else:
                 # Flattening the list of courses and get unique course IDs
                final_recommended_list = list(set([course for course in recommended_list]))

                with recommendation_container:
                    for course_id in final_recommended_list:
                        # Check if user has already taken the course
                        if ((user_type == 'Existing User') &  ((df_encoded['course_id'] == course_id) & (df_encoded['user_id'] == target_user_id)).any()):
                            continue
                        else:
                            # Get course details
                            course_info = df_encoded[df_encoded['course_id'] == course_id][['course_name', 'difficulty_level','rating']].iloc[0]
                            course_name = course_info['course_name']
                            difficulty_level = course_info['difficulty_level']
                            rating = course_info['rating']
                    
                            # Display course information in a nice format
                            col1, col2, col3  = st.columns([1, 2, 1])
                            with col1:
                                st.write(f"**Course ID:** {course_id}")
                            with col2:
                                st.write(f"**Course Name:** {course_name}")
                            with col3:
                                st.write(f"**Difficulty:** {difficulty_level}")
                            st.divider()

# Add some instructions at the bottom
st.sidebar.markdown("---")
st.sidebar.info("""
    **How to use this app:**
    1. Select a user ID from the dropdown
    2. Click "Generate Recommendations"
    3. View personalized course recommendations
""")

# You can run this app with: streamlit run recommendation_app.py