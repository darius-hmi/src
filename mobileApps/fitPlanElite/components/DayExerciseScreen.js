import React, { useEffect, useState, useContext } from 'react';
import { View, Text, TextInput, Button, Alert, StyleSheet, ActivityIndicator, FlatList, KeyboardAvoidingView, Platform } from 'react-native';
import { UserContext } from '../App'; // Adjust the path based on your folder structure
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Picker } from '@react-native-picker/picker'; // Import Picker component

const DayExerciseScreen = ({ route }) => {
  const { date, muscleGroups, allExercises } = route.params; // Receive date, muscleGroups, and allExercises from navigation
  const [loading, setLoading] = useState(true);
  const [exercises, setExercises] = useState([]);
  const [selectedMuscleGroup, setSelectedMuscleGroup] = useState('');
  const [selectedExercise, setSelectedExercise] = useState('');
  const [weight, setWeight] = useState('');
  const [sets, setSets] = useState('');
  const [reps, setReps] = useState('');
  const [showForm, setShowForm] = useState(false); // State to control form visibility
  const { user } = useContext(UserContext);

  useEffect(() => {
    const fetchExercises = async () => {
      try {
        const accessToken = await AsyncStorage.getItem('accessToken');
        const response = await fetch(`http://192.168.0.192:8000/api/exercises/?date=${date}`, {
          headers: {
            'Authorization': `Bearer ${accessToken}`,
          },
        });

        if (!response.ok) {
          throw new Error('Failed to fetch exercises');
        }

        const data = await response.json();
        setExercises(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching exercises:', error);
        setLoading(false);
      }
    };

    fetchExercises();
  }, [date]);

  const handleAddExercise = async () => {
    if (!selectedExercise || !selectedMuscleGroup || !weight || !sets || !reps) {
      Alert.alert('Error', 'Please enter all fields.');
      return;
    }

    const formData = {
      name: selectedExercise,
      muscle: selectedMuscleGroup,
      weight: parseInt(weight),
      sets: parseInt(sets),
      reps: parseInt(reps),
      author: user.id,
      created_at: date // Set the created_at field to the selected date
    };

    try {
      const accessToken = await AsyncStorage.getItem('accessToken');
      const response = await fetch('http://192.168.0.192:8000/api/exercises/', {
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

      const newExercise = await response.json();
      setExercises([...exercises, newExercise]); // Add the new exercise to the list of exercises
      Alert.alert('Success', 'Exercise added successfully!');
      setSelectedExercise('');
      setSelectedMuscleGroup('');
      setWeight('');
      setSets('');
      setReps('');
      setShowForm(false); // Close the form after adding the exercise
    } catch (error) {
      console.error('Error adding exercise:', error);
      Alert.alert('Error', 'Failed to add exercise.');
    }
  };

  const handleCancel = () => {
    setShowForm(false);
    setSelectedExercise('');
    setSelectedMuscleGroup('');
    setWeight('');
    setSets('');
    setReps('');
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
        <Text style={styles.dateTitle}>Exercises for {date}</Text>
      </View>

      <FlatList
        style={styles.flatList}
        contentContainerStyle={styles.contentContainer}
        data={exercises}
        keyExtractor={(item) => item.id.toString()}
        renderItem={({ item }) => (
          <View style={styles.exerciseItem}>
            <Text>{item.name}</Text>
            <Text>Weight: {item.weight}</Text>
            <Text>Sets: {item.sets}</Text>
            <Text>Reps: {item.reps}</Text>
          </View>
        )}
      />

      {showForm && (
        <View style={styles.formContainer}>
          <Text style={styles.label}>Muscle Group</Text>
          <Picker
            selectedValue={selectedMuscleGroup}
            onValueChange={(itemValue) => setSelectedMuscleGroup(itemValue)}
            style={styles.input}
          >
            <Picker.Item label="Select Muscle Group" value="" />
            {muscleGroups.map((group, index) => (
              <Picker.Item key={index} label={group} value={group} />
            ))}
          </Picker>

          <Text style={styles.label}>Exercise Name</Text>
          <Picker
            selectedValue={selectedExercise}
            onValueChange={(itemValue) => setSelectedExercise(itemValue)}
            style={styles.input}
          >
            <Picker.Item label="Select Exercise" value="" />
            {(allExercises[selectedMuscleGroup] || []).map((exercise, index) => (
              <Picker.Item key={index} label={exercise} value={exercise} />
            ))}
          </Picker>

          <Text style={styles.label}>Weight</Text>
          <TextInput
            style={styles.input}
            placeholder="Weight"
            keyboardType="numeric"
            value={weight}
            onChangeText={setWeight}
            onSubmitEditing={() => {}}
          />

          <Text style={styles.label}>Sets</Text>
          <TextInput
            style={styles.input}
            placeholder="Sets"
            keyboardType="numeric"
            value={sets}
            onChangeText={setSets}
            onSubmitEditing={() => {}}
          />

          <Text style={styles.label}>Reps</Text>
          <TextInput
            style={styles.input}
            placeholder="Reps"
            keyboardType="numeric"
            value={reps}
            onChangeText={setReps}
            onSubmitEditing={() => {}}
          />

          <View style={styles.buttonContainer}>
            <Button title="Add New Exercise" onPress={handleAddExercise} />
            <Button title="Cancel" color="red" onPress={handleCancel} />
          </View>
        </View>
      )}

      {!showForm && (
        <View style={styles.showFormButtonContainer}>
          <Button title="Add New Exercise" onPress={() => setShowForm(true)} />
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
  exerciseItem: {
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

export default DayExerciseScreen;
