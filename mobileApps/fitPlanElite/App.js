import 'react-native-gesture-handler';
import 'react-native-reanimated';
import React, { createContext, useState, useContext, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { View, TextInput, Text, Button, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import RemindersScreen from './components/RemindersScreen';
import BudgetScreen from './components/BudgetScreen';
import ExerciseScreen from './components/ExerciseScreen';
import DayExerciseScreen from './components/DayExerciseScreen';
import FoodScreen from './components/FoodScreen';
import DayFoodScreen from './components/DayFoodScreen';

// Create a context for the user
export const UserContext = createContext();

// Define screen components
const HomeScreen = ({ navigation }) => {
  const { user } = useContext(UserContext);

  // Optional: Handle loading state while user data is being fetched
  if (!user) {
    return (
      <View style={styles.screenContainer}>
        <Text>Loading...</Text>
      </View>
    );
  }

  const navigateTo = (screenName) => {
    navigation.navigate(screenName);
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.box} onPress={() => navigateTo('Exercise')}>
        <Text style={styles.text}>Exercise Hub</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.box} onPress={() => navigateTo('Food')}>
        <Text style={styles.text}>Food Hub</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.box} onPress={() => navigateTo('Reminders')}>
        <Text style={styles.text}>Reminders</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.box} onPress={() => navigateTo('Budget')}>
        <Text style={styles.text}>Budget</Text>
      </TouchableOpacity>
    </View>
  );
};

const LoginScreen = ({ navigation }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const { setUser } = useContext(UserContext);

  const handleLogin = async () => {
    try {
      const response = await fetch('http://192.168.0.192:8000/api/token/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });
      
      if (response.status === 200) {
        const data = await response.json();
        // Store tokens, username, and user ID in AsyncStorage
        await AsyncStorage.setItem('accessToken', data.access);
        await AsyncStorage.setItem('refreshToken', data.refresh);

        // Decode the access token to get the user ID
        const decodedToken = JSON.parse(atob(data.access.split('.')[1]));
        const userId = decodedToken.user_id;

        // Store username and user ID in AsyncStorage
        await AsyncStorage.setItem('username', username);
        await AsyncStorage.setItem('userId', userId.toString()); // Store user ID as string

        setUser({ username, id: userId });
        navigation.navigate('Home');
      } else {
        Alert.alert('Login failed', 'Invalid username or password');
      }
    } catch (error) {
      Alert.alert('Error', 'Something went wrong. Please try again.');
    }
  };

  return (
    <View style={styles.screenContainer}>
      <Text style={styles.label}>Username</Text>
      <TextInput
        style={styles.input}
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
      />
      <Text style={styles.label}>Password</Text>
      <TextInput
        style={styles.input}
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />
      <Button title="Login" onPress={handleLogin} />
    </View>
  );
};

const Stack = createStackNavigator();

const App = () => {
  const [user, setUser] = useState(null);

  // Check AsyncStorage for username on app start
  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const username = await AsyncStorage.getItem('username');
        const userId = await AsyncStorage.getItem('userId');
        if (username && userId) {
          setUser({ username, id: parseInt(userId) });
        }
      } catch (error) {
        console.log('Error fetching user data from AsyncStorage:', error);
      }
    };
  
    fetchUserData();
  }, []);

  return (
    <UserContext.Provider value={{ user, setUser }}>
      <NavigationContainer>
        <Stack.Navigator initialRouteName="Login">
          <Stack.Screen name="Login" component={LoginScreen} />
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen name="Exercise" component={ExerciseScreen} />
          <Stack.Screen name="Food" component={FoodScreen} />
          <Stack.Screen name="Reminders" component={RemindersScreen} />
          <Stack.Screen name="Budget" component={BudgetScreen} />
          <Stack.Screen name="DayExercise" component={DayExerciseScreen} />
          <Stack.Screen name="DayFood" component={DayFoodScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </UserContext.Provider>
  );
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
    alignItems: 'center',
    padding: 20,
  },
  header: {
    position: 'absolute',
    top: 20,
    left: 20,
  },
  headerText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  box: {
    width: '45%',
    aspectRatio: 1, // Makes the boxes square
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
    marginBottom: 20,
  },
  text: {
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  screenContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  label: {
    fontSize: 18,
    marginBottom: 10,
  },
  input: {
    height: 40,
    borderColor: '#ccc',
    borderWidth: 1,
    marginBottom: 20,
    paddingHorizontal: 10,
    width: '80%',
  },
});

export default App;
