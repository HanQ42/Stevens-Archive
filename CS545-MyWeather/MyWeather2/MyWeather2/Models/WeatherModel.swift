//
//  WeatherModel.swift
//  MyWeather2
//
//  Created by Hanqing Liu on 12/2/20.
//

import Foundation

// Weather Model in Json Returned
struct WeatherModel {
    let conditionId: Int
    let cityName: String
    let description : String
    let temperature: Double
    let windSpeed: Double
    let WindDegree: Double
    let pressure: Int
    let humidity: Int
    let cloudiness: Int
    let visibility: Int
}
