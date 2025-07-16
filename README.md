# 🚢 Battleship Agent

A reinforcement learning agent trained to play Battleship using Q-learning.

## 🎯 Overview

This project implements an intelligent Battleship agent that learns to play optimally through Q-learning. The agent can play against other agents or human players using a tuple space communication system.

## 🏗️ Architecture

### Core Classes
- **🚢 Ship**: Represents individual ships with hit tracking and positioning
- **⚔️ Battleship**: Game environment managing grids, ships, and actions
- **🤖 BattleshipAgent**: AI agent with Q-learning capabilities and communication

### Key Features
- **📊 Q-learning**: Epsilon-greedy policy with experience-based learning
- **🔄 Multi-agent**: Two agents communicate via tuple space
- **👤 Human vs AI**: Interactive gameplay option
- **📈 Training**: 40,000 episodes with reward optimization

## 🎮 Game Parameters

- **Grid Size**: 4x4 (expandable to 7x7)
- **Ships**: Multiple ships with no diagonal contact
- **Rewards**: Hit (+150), Sunk (+30), Win (+1030)
- **Penalties**: Miss (-25), Already hit (-200), Turn (-50)

## 🚀 Training Results

The agent achieved improved performance through:
- ✅ Increased training episodes (40K)
- ✅ Optimized reward structure
- ✅ Enhanced penalty system
- ✅ Consecutive hit/miss handling

## 🔧 Usage

Agents communicate through tuple space messages:
- `("Request", counter, x, y)` - Attack request
- `("Response", counter, x, y, value)` - Attack result
- `("Game over", counter, result)` - Game end signal

## 🎯 Future Improvements

- 🔍 Efficiency optimization for larger grids
- 📚 Better Q-table representation
- 🎲 Enhanced ship positioning strategies
- 🎨 Improved human vs agent interface

---

*Developed by Casali Cristian for the course of Distributed Artificial Intelligence, feb 2025*
