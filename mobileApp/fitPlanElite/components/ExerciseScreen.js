import React, { useState, useEffect, useContext } from 'react';
import { View, Text, TextInput, Button, Alert, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { Calendar } from 'react-native-calendars';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { UserContext } from '../App'; // Adjust the path based on your folder structure

const ExerciseScreen = ({ navigation }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [exercises, setExercises] = useState([]);
  const [newExercise, setNewExercise] = useState('');
  const [newMuscleGroup, setNewMuscleGroup] = useState('');
  const [weight, setWeight] = useState('');
  const [sets, setSets] = useState('');
  const [reps, setReps] = useState('');
  const { user } = useContext(UserContext);
  const [token, setToken] = useState(null);

  const muscleGroups = ['Chest', 'Shoulder', 'Legs', 'Back', 'Arms'];
  const allExercises = {
    'Chest': ['Flat barbell bench press', 'Incline barbell bench press', 'Flat dumbbell Bench Press', 'Incline dumbbell press low angle', 'Incline dumbbell press high angle', 'Dumbbell flyes', 'Cable flyes', 'Machine flyes', 'Machine Chest Press'],
    'Shoulder': ['Dumbbell shoulder press', 'Barbell Dumbbell press', 'Cable lateral raises', 'Dumbbell lateral raises', 'Dumbbell Front Raises', 'Cable Front Raises', 'Front Raises', 'Rear Delt Machine', 'Rear Delt Dumbbell', 'Machine shoulder press', 'Traps raises'],
    'Legs': ['Squats', 'Leg press', 'Leg extension', 'Leg curls', 'Bulgarian split squats', 'Romanian deadlift'],
    'Back': ['Barbell Bent over row', 'Rope Pulldown', 'Deadlift', 'Lat pulldown straight bar', 'Lat pulldown close grip', 'Cable row wide', 'Cable row narrow', 'Single arm lat pulldown', 'Machine Row', 'pull ups'],
    'Arms': ['Bicep curls', 'Hammer curls', 'Cable bicep curls', 'Cable hammer curl', 'Cable tricep extension', 'Bar tricep extension', 'Overhead extension', 'Dips', 'underhand lat pulldown']
  };

  useEffect(() => {
    const fetchToken = async () => {
      const accessToken = await AsyncStorage.getItem('accessToken');
      setToken(accessToken);
      fetchExercises(accessToken);
    };
    fetchToken();
  }, []);

  const fetchExercises = async (accessToken) => {
    try {
      const response = await fetch('http://192.168.0.192:8000/api/exercises/', {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      setExercises(data);
      setLoading(false);
    } catch (error) {
      setError(error);
      setLoading(false);
    }
  };

  const handleAddExercise = async () => {
    if (!newExercise || !newMuscleGroup || !weight || !sets || !reps) {
      Alert.alert('Error', 'Please enter all fields.');
      return;
    }

    const formData = {
      name: newExercise,
      muscle: newMuscleGroup,
      weight: parseInt(weight),
      sets: parseInt(sets),
      reps: parseInt(reps),
      author: user.id, // Use the logged-in user's ID
    };

    try {
      const response = await fetch('http://192.168.0.192:8000/api/exercises/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        Alert.alert('Success', 'Exercise added successfully!');
        fetchExercises(token); // Refresh the exercises list
        setNewExercise('');
        setNewMuscleGroup('');
        setWeight('');
        setSets('');
        setReps('');
      } else {
        Alert.alert('Error', 'Failed to add exercise.');
      }
    } catch (error) {
      console.error('Error adding exercise:', error);
      Alert.alert('Error', 'Failed to add exercise.');
    }
  };

  const handleDayPress = (day) => {
    navigation.navigate('DayExercise', { date: day.dateString, muscleGroups, allExercises });
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
      <Text style={styles.label}>Muscle Group</Text>
      <Picker
        selectedValue={newMuscleGroup}
        onValueChange={(itemValue) => {
          setNewMuscleGroup(itemValue);
          setNewExercise('');
        }}
        style={styles.input}
      >
        <Picker.Item label="Select Muscle Group" value="" />
        {muscleGroups.map((group, index) => (
          <Picker.Item key={index} label={group} value={group} />
        ))}
      </Picker>

      <Text style={styles.label}>Exercise Name</Text>
      <Picker
        selectedValue={newExercise}
        onValueChange={(itemValue) => setNewExercise(itemValue)}
        style={styles.input}
      >
        <Picker.Item label="Select Exercise" value="" />
        {(allExercises[newMuscleGroup] || []).map((exercise, index) => (
          <Picker.Item key={index} label={exercise} value={exercise} />
        ))}
      </Picker>

      <Text style={styles.label}>Weight</Text>
      <TextInput
        style={styles.input}
        placeholder="Weight"
        value={weight}
        onChangeText={setWeight}
        keyboardType="numeric"
      />

      <Text style={styles.label}>Sets</Text>
      <TextInput
        style={styles.input}
        placeholder="Sets"
        value={sets}
        onChangeText={setSets}
        keyboardType="numeric"
      />

      <Text style={styles.label}>Reps</Text>
      <TextInput
        style={styles.input}
        placeholder="Reps"
        value={reps}
        onChangeText={setReps}
        keyboardType="numeric"
      />

      <Button title="Add Exercise" onPress={handleAddExercise} />

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

export default ExerciseScreen;
