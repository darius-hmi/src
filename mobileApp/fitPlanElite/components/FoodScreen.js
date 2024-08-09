import React, { useState, useEffect, useContext } from 'react';
import { View, Text, TextInput, Button, Alert, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Picker } from '@react-native-picker/picker';
import { Calendar } from 'react-native-calendars';
import { UserContext } from '../App'; // Adjust the path based on your folder structure

const FoodScreen = ({ navigation }) => {
  const { user } = useContext(UserContext);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [foodName, setFoodName] = useState('');
  const [calories, setCalories] = useState('');
  const [protein, setProtein] = useState('');
  const [token, setToken] = useState(null);

  useEffect(() => {
    const fetchToken = async () => {
      const accessToken = await AsyncStorage.getItem('accessToken');
      setToken(accessToken);
    };
    fetchToken();
  }, []);

  const handleAddFood = async () => {
    if (!foodName || !calories || !protein) {
      Alert.alert('Error', 'Please enter all fields.');
      return;
    }

    setLoading(true);

    const formData = {
      name: foodName,
      calories: parseInt(calories),
      protein: parseInt(protein),
      author: user.id // Assuming user.id is required for the backend
    };

    try {
      const response = await fetch('http://192.168.0.192:8000/api/foods/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      Alert.alert('Success', 'Food added successfully!');
      setFoodName('');
      setCalories('');
      setProtein('');
    } catch (error) {
      console.error('Error adding food:', error);
      Alert.alert('Error', 'Failed to add food.');
      setError(error);
    } finally {
      setLoading(false);
    }
  };

  const handleDayPress = (day) => {
    navigation.navigate('DayFood', { date: day.dateString });
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0000ff" />
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.container}>
        <Text>Error: {error.message}</Text>
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.label}>Food Name</Text>
      <TextInput
        style={styles.input}
        placeholder="Food Name"
        value={foodName}
        onChangeText={setFoodName}
      />

      <Text style={styles.label}>Calories</Text>
      <TextInput
        style={styles.input}
        placeholder="Calories"
        keyboardType="numeric"
        value={calories}
        onChangeText={setCalories}
      />

      <Text style={styles.label}>Protein (g)</Text>
      <TextInput
        style={styles.input}
        placeholder="Protein"
        keyboardType="numeric"
        value={protein}
        onChangeText={setProtein}
      />

      <Button title="Add Food" onPress={handleAddFood} />

      <Text style={styles.calendarTitle}>Calendar View</Text>
      <Text style={styles.calendarSubtitle}>Click on any day to view</Text>
      
      <Calendar
        onDayPress={handleDayPress}
        // Other configurations
      />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 20,
  },
  label: {
    fontSize: 18,
    fontWeight: 'bold',
    marginVertical: 10,
  },
  input: {
    height: 50,
    borderColor: 'gray',
    borderWidth: 1,
    marginBottom: 10,
    paddingHorizontal: 10,
    justifyContent: 'center',
  },
  calendarTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 10,
    textAlign: 'center',
  },
  calendarSubtitle: {
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
  },
});

export default FoodScreen;
