import React, { useState, useEffect, useContext } from 'react';
import { View, Text, FlatList, ActivityIndicator, StyleSheet, TextInput, Button, Alert, TouchableOpacity} from 'react-native';
import { Swipeable } from 'react-native-gesture-handler';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { UserContext } from '../App'; // Adjust the path based on your folder structure

const RemindersScreen = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [todoLists, setTodoLists] = useState([]);
  const [newTask, setNewTask] = useState('');
  const [newDueDate, setNewDueDate] = useState('');
  const { user } = useContext(UserContext);
  const [token, setToken] = useState(null);

  useEffect(() => {
    const fetchToken = async () => {
      const accessToken = await AsyncStorage.getItem('accessToken');
      setToken(accessToken);
      fetchToDoList(accessToken);
    };
    fetchToken();
  }, []);

  const fetchToDoList = async (accessToken) => {
    try {
      const response = await fetch('http://192.168.0.192:8000/api/todolists/', {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      setTodoLists(data);
      setLoading(false);
    } catch (error) {
      setError(error);
      setLoading(false);
    }
  };

  const handleAddTask = async () => {
    if (!newTask || !newDueDate) {
      Alert.alert('Error', 'Please enter both task and due date.');
      return;
    }

    const formData = {
      task: newTask,
      dueDate: newDueDate,
      author: user.id, // Use the logged-in user's ID
    };

    try {
      const response = await fetch('http://192.168.0.192:8000/api/todolists/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        Alert.alert('Success', 'Task added successfully!');
        fetchToDoList(token); // Refresh the ToDo list
        setNewTask('');
        setNewDueDate('');
      } else {
        Alert.alert('Error', 'Failed to add task.');
      }
    } catch (error) {
      console.error('Error adding task:', error);
      Alert.alert('Error', 'Failed to add task.');
    }
  };

  const handleDeleteToDoLists = async (id) => {
    try {
      const response = await fetch(`http://192.168.0.192:8000/api/todolists/${id}/`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
      });

      if (response.ok) {
        Alert.alert('Success', 'Item deleted successfully!');
        fetchToDoList(token); // Refresh the expense list
      } else {
        Alert.alert('Error', 'Failed to delete item.');
      }
    } catch (error) {
      console.error('Error deleting item:', error);
      Alert.alert('Error', 'Failed to delete item.');
    }
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
    <View style={styles.container}>
      <FlatList
        data={todoLists}
        keyExtractor={(item) => item.id.toString()}
        renderItem={({ item }) => (
          <Swipeable
            renderRightActions={() => (
              <TouchableOpacity
                style={styles.deleteButton}
                onPress={() => handleDeleteToDoLists(item.id)}
              >
                <Text style={styles.deleteText}>Delete</Text>
              </TouchableOpacity>
            )}
          >
            <View style={styles.item}>
              <Text style={styles.title}>{item.task}</Text>
              <Text style={styles.dueDate}>Due: {item.dueDate}</Text>
            </View>
          </Swipeable>
        )}
      />

      <TextInput
        style={styles.input}
        placeholder="Task"
        value={newTask}
        onChangeText={setNewTask}
      />
      <TextInput
        style={styles.input}
        placeholder="Due Date (YYYY-MM-DD)"
        value={newDueDate}
        onChangeText={setNewDueDate}
      />
      <Button title="Add Task" onPress={handleAddTask} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  item: {
    marginBottom: 10,
    padding: 10,
    backgroundColor: '#f9c2ff',
    borderRadius: 5,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  dueDate: {
    fontSize: 14,
    color: '#555',
  },
  input: {
    height: 40,
    borderColor: 'gray',
    borderWidth: 1,
    marginBottom: 10,
    paddingHorizontal: 10,
  },
  deleteButton: {
    backgroundColor: 'red',
    justifyContent: 'center',
    alignItems: 'center',
    width: 80,
    height: '90%',
    borderRadius: 5,
  },
  deleteText: {
    color: 'white',
    fontWeight: 'bold',
  },
});

export default RemindersScreen;
