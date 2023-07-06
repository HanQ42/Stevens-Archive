//
//  WeatherJson.swift
//  MyWeather2
//
//  Created by Hanqing Liu on 12/2/20.
//

import Foundation

// Json that returned from Open Weather API
struct WeatherData: Codable {
    let name: String
    let main: Main
    let wind: Wind
    let clouds: Clouds
    let weather: [Weather]
    let visibility: Int
}

struct Main: Codable {
    let temp: Double
    let pressure: Int
    let humidity: Int
}

struct Wind: Codable {
    let speed: Double
    let deg: Double
}

struct Clouds: Codable {
    let all: Int
}

struct Weather: Codable {
    let description: String
    let id: Int
}
