import React, { useEffect, useState, useContext } from 'react';
import { View, Text, TextInput, Button, Alert, StyleSheet, ActivityIndicator, FlatList, KeyboardAvoidingView, Platform } from 'react-native';
import { UserContext } from '../App'; // Adjust the path based on your folder structure
import AsyncStorage from '@react-native-async-storage/async-storage';

const DayFoodScreen = ({ route }) => {
  const { date } = route.params;
  const [loading, setLoading] = useState(true);
  const [foods, setFoods] = useState([]);
  const [foodName, setFoodName] = useState('');
  const [calories, setCalories] = useState('');
  const [protein, setProtein] = useState('');
  const [showForm, setShowForm] = useState(false); // State to control form visibility
  const { user } = useContext(UserContext);

  useEffect(() => {
    const fetchFoods = async () => {
      try {
        const accessToken = await AsyncStorage.getItem('accessToken');
        const response = await fetch(`http://192.168.0.192:8000/api/foods/?date=${date}`, {
          headers: {
            'Authorization': `Bearer ${accessToken}`,
          },
        });

        if (!response.ok) {
          throw new Error('Failed to fetch foods');
        }

        const data = await response.json();
        setFoods(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching foods:', error);
        setLoading(false);
      }
    };

    fetchFoods();
  }, [date]);

  const handleAddFood = async () => {
    if (!foodName || !calories || !protein) {
      Alert.alert('Error', 'Please enter all fields.');
      return;
    }

    const formData = {
      name: foodName,
      calories: parseInt(calories),
      protein: parseInt(protein),
      author: user.id,
      created_at: date // Set the created_at field to the selected date
    };

    try {
      const accessToken = await AsyncStorage.getItem('accessToken');
      const response = await fetch('http://192.168.0.192:8000/api/foods/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const newFood = await response.json();
      setFoods([...foods, newFood]); // Add the new food to the list of foods
      Alert.alert('Success', 'Food added successfully!');
      setFoodName('');
      setCalories('');
      setProtein('');
      setShowForm(false); // Close the form after adding the food
    } catch (error) {
      console.error('Error adding food:', error);
      Alert.alert('Error', 'Failed to add food.');
    }
  };

  const handleCancel = () => {
    setShowForm(false);
    setFoodName('');
    setCalories('');
    setProtein('');
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0000ff" />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
    >
      <View style={styles.headerContainer}>
        <Text style={styles.dateTitle}>Foods consumed on {date}</Text>
      </View>

      <FlatList
        style={styles.flatList}
        contentContainerStyle={styles.contentContainer}
        data={foods}
        keyExtractor={(item) => item.id.toString()}
        renderItem={({ item }) => (
          <View style={styles.foodItem}>
            <Text>{item.name}</Text>
            <Text>Calories: {item.calories}</Text>
            <Text>Protein: {item.protein}</Text>
          </View>
        )}
      />

      {showForm && (
        <View style={styles.formContainer}>
          <Text style={styles.label}>Food Name</Text>
          <TextInput
            style={styles.input}
            placeholder="Food Name"
            value={foodName}
            onChangeText={setFoodName}
            onSubmitEditing={() => {}}
          />

          <Text style={styles.label}>Calories</Text>
          <TextInput
            style={styles.input}
            placeholder="Calories"
            keyboardType="numeric"
            value={calories}
            onChangeText={setCalories}
            onSubmitEditing={() => {}}
          />

          <Text style={styles.label}>Protein (g)</Text>
          <TextInput
            style={styles.input}
            placeholder="Protein"
            keyboardType="numeric"
            value={protein}
            onChangeText={setProtein}
            onSubmitEditing={() => {}}
          />

          <View style={styles.buttonContainer}>
            <Button title="Add New Food" onPress={handleAddFood} />
            <Button title="Cancel" color="red" onPress={handleCancel} />
          </View>
        </View>
      )}

      {!showForm && (
        <View style={styles.showFormButtonContainer}>
          <Button title="Add New Food" onPress={() => setShowForm(true)} />
        </View>
      )}

    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerContainer: {
    padding: 20,
    backgroundColor: '#f0f0f0',
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
  },
  flatList: {
    flex: 1,
    backgroundColor: '#fff',
  },
  contentContainer: {
    paddingHorizontal: 20,
  },
  dateTitle: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  foodItem: {
    paddingVertical: 10,
    borderBottomColor: '#ccc',
    borderBottomWidth: 1,
  },
  formContainer: {
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: '#ccc',
  },
  label: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  input: {
    height: 40,
    borderColor: 'gray',
    borderWidth: 1,
    marginBottom: 10,
    paddingHorizontal: 10,
  },
  showFormButtonContainer: {
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: '#ccc',
    alignItems: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 20,
  },
});

export default DayFoodScreen;
