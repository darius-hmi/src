import React, { useState, useEffect, useContext } from 'react';
import { View, Text, FlatList, ActivityIndicator, StyleSheet, TextInput, Button, Alert, TouchableOpacity } from 'react-native';
import { Swipeable } from 'react-native-gesture-handler';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { UserContext } from '../App'; // Adjust the import based on your file structure

const BudgetScreen = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expenses, setExpenses] = useState([]);
  const [newExpenseName, setNewExpenseName] = useState('');
  const [newExpenseAmount, setNewExpenseAmount] = useState('');
  const [newExpenseDate, setNewExpenseDate] = useState('');
  const { user } = useContext(UserContext);
  const [token, setToken] = useState(null);

  useEffect(() => {
    const fetchToken = async () => {
      const accessToken = await AsyncStorage.getItem('accessToken');
      setToken(accessToken);
      fetchExpenses(accessToken);
    };

    fetchToken();
  }, []);

  const fetchExpenses = async (accessToken) => {
    try {
      const response = await fetch('http://192.168.0.192:8000/api/expenses/', {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      setExpenses(data);
      setLoading(false);
    } catch (error) {
      setError(error);
      setLoading(false);
    }
  };

  const handleAddExpense = async () => {
    if (!newExpenseName || !newExpenseAmount || !newExpenseDate) {
      Alert.alert('Error', 'Please enter all fields.');
      return;
    }

    const formData = {
      name: newExpenseName,
      amount: parseFloat(newExpenseAmount), // Ensure it's in the correct format
      date: parseInt(newExpenseDate), // Ensure it's in the correct format
      user: user.id
    };

    try {
      const response = await fetch('http://192.168.0.192:8000/api/expenses/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        Alert.alert('Success', 'Expense added successfully!');
        fetchExpenses(token); // Refresh the expense list
        setNewExpenseName('');
        setNewExpenseAmount('');
        setNewExpenseDate('');
      } else {
        Alert.alert('Error', 'Failed to add expense.');
      }
    } catch (error) {
      console.error('Error adding expense:', error);
      Alert.alert('Error', 'Failed to add expense.');
    }
  };

  const handleDeleteExpense = async (id) => {
    try {
      const response = await fetch(`http://192.168.0.192:8000/api/expenses/${id}/`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
      });

      if (response.ok) {
        Alert.alert('Success', 'Expense deleted successfully!');
        fetchExpenses(token); // Refresh the expense list
      } else {
        Alert.alert('Error', 'Failed to delete expense.');
      }
    } catch (error) {
      console.error('Error deleting expense:', error);
      Alert.alert('Error', 'Failed to delete expense.');
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
        data={expenses}
        keyExtractor={(item) => item.id.toString()}
        renderItem={({ item }) => (
          <Swipeable
            renderRightActions={() => (
              <TouchableOpacity
                style={styles.deleteButton}
                onPress={() => handleDeleteExpense(item.id)}
              >
                <Text style={styles.deleteText}>Delete</Text>
              </TouchableOpacity>
            )}
          >
            <View style={styles.item}>
              <Text style={styles.title}>{item.name}</Text>
              <Text style={styles.amount}>Amount: {item.amount}</Text>
              <Text style={styles.date}>Date: {item.date}</Text>
            </View>
          </Swipeable>
        )}
      />

      <TextInput
        style={styles.input}
        placeholder="Expense Name"
        value={newExpenseName}
        onChangeText={setNewExpenseName}
      />
      <TextInput
        style={styles.input}
        placeholder="Amount"
        value={newExpenseAmount}
        onChangeText={setNewExpenseAmount}
        keyboardType="numeric"
      />
      <TextInput
        style={styles.input}
        placeholder="Date (YYYY)"
        value={newExpenseDate}
        onChangeText={setNewExpenseDate}
        keyboardType="numeric"
      />
      <Button title="Add Expense" onPress={handleAddExpense} />
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
  amount: {
    fontSize: 14,
    color: '#555',
  },
  date: {
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

export default BudgetScreen;