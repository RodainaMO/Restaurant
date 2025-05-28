import os
import sys
import streamlit as st
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.recommender import ZomatoRecommender

st.set_page_config(
    page_title="Zomato Restaurant Recommender",
    layout="wide",
    page_icon="🍽️"
)

# --- Load Data with Caching ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/cleaned_zomato.csv')

        # Standardize column names
        df.columns = df.columns.str.strip().str.title()

        # Ensure required columns exist
        required_cols = ['Restaurant Name', 'City', 'Primary Cuisine', 
                         'Cost Category', 'Rating', 'Votes','Longitude', 'Latitude']
        if not all(col in df.columns for col in required_cols):
            st.error("Missing required columns in the dataset")
            st.write("Available columns:", df.columns.tolist())  # Optional debug
            return None

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# --- Initialize App ---
def initialize_app():
    data = load_data()
    if data is None:
        st.stop()
    return ZomatoRecommender(data)

recommender = initialize_app()

def display_restaurant_card(restaurant):
    """Display a restaurant card with location handling"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(restaurant['Restaurant Name'])
            
            st.markdown(f"""
            **🍽️ Cuisine:** {restaurant['Primary Cuisine'].title()}  
            **💰 Price Range:** {restaurant['Cost Category'].title()}  
            **⭐ Rating:** {restaurant['Rating']} ({int(restaurant['Votes'])} votes)
            """)
            
            with st.expander("📍 Location Details", expanded=False):
                # Always show city
                st.markdown(f"**🏙️ City:** {restaurant['City'].title()}")
                
                # Clean coordinate handling
                if all(key in restaurant for key in ['Latitude', 'Longitude']):
                    try:
                        lat = float(restaurant['Latitude'])
                        lon = float(restaurant['Longitude'])
                        
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            maps_link = f"https://www.google.com/maps?q={lat},{lon}"
                            st.markdown(f"[🗺️ Open in Google Maps]({maps_link})")
                            st.caption(f"Coordinates: {lat:.6f}, {lon:.6f}")
                        else:
                            st.warning("Location coordinates out of valid range")
                    except (ValueError, TypeError):
                        st.info("Could not display map link due to invalid coordinates")
                else:
                    st.info("Detailed location data not available")
                
                

        with col2:
            st.markdown(f"**Score:** {restaurant['Score']:.2f}")
        
        st.markdown("---")
# --- Main App ---
def main():
    st.title("🍽️ Zomato Restaurant Recommender")
    st.markdown("""
    Find perfect dining spots based on your preferences.  
    Adjust filters in the sidebar and click "Find Restaurants".
    """)

    # --- Sidebar Filters ---
    with st.sidebar:
        st.header("🔍 Filter Preferences")
        
        # Location
        all_locations = sorted(recommender.data['City'].str.title().unique())
        selected_location = st.selectbox(
            "Select Location",
            all_locations,
            index=0,
            help="Choose your preferred city/area"
        )

        # Cuisine
        all_cuisines = sorted(recommender.data['Primary Cuisine'].str.title().unique())
        selected_cuisines = st.multiselect(
            "Select Cuisine(s)",
            all_cuisines,
            default=[],
            help="Select one or more cuisine types"
        )

        # Budget
        budget_mapping = {
            "Low (Under ₹300)": "low",
            "Medium (₹300-₹600)": "medium",
            "High (Above ₹600)": "high"
        }
        selected_budget = st.selectbox(
            "Select Budget Range",
            list(budget_mapping.keys()),
            index=1
        )
        budget_value = budget_mapping[selected_budget]

        # Additional Filters
        with st.expander("Advanced Options"):
            min_rating = st.slider(
                "Minimum Rating",
                min_value=1.0,
                max_value=5.0,
                value=3.5,
                step=0.1
            )
            min_votes = st.slider(
                "Minimum Votes (Popularity)",
                min_value=0,
                max_value=1000,
                value=50,
                step=10
            )
            num_results = st.slider(
                "Number of Results",
                min_value=1,
                max_value=20,
                value=10
            )

        search_button = st.button("Find Restaurants", type="primary")

    # --- Recommendation Display ---
    if search_button:
        with st.spinner("Finding the best matches..."):
            try:
                recommendations = recommender.recommend(
                    cuisines=[c.lower() for c in selected_cuisines],
                    budget_range=(budget_value, budget_value),
                    location=selected_location.lower(),
                    top_n=num_results
                )

                # Apply additional filters
                if not recommendations.empty:
                    recommendations = recommendations[
                        (recommendations['Rating'] >= min_rating) &
                        (recommendations['Votes'] >= min_votes)
                    ]

                if recommendations.empty:
                    st.warning("""
                    No matching restaurants found.  
                    Try adjusting your filters (especially location or cuisine).
                    """)
                else:
                    st.success(f"Found {len(recommendations)} matching restaurants")
                    
                    # Display results
                    st.subheader("🍴 Top Recommendations")
                    for _, row in recommendations.iterrows():
                        display_restaurant_card(row)
                        
                    # Download option
                    st.download_button(
                        label="Download Recommendations",
                        data=recommendations.to_csv(index=False),
                        file_name="zomato_recommendations.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()