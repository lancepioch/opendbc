from opendbc.car.structs import CarControlSP
import itertools
import math
import unittest

import numpy as np

from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.lateral import get_max_angle_vm
from opendbc.car.subaru.carcontroller import get_safety_CP
from opendbc.car.subaru.carstate import CarState
from opendbc.car.subaru.fingerprints import FW_VERSIONS
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR, CarControllerParams, SubaruFlags
from opendbc.car.vehicle_model import VehicleModel


class TestSubaruFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    for platform, fws_per_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fws in fws_per_ecu.items():
        fw_size = len(fws[0])
        for fw in fws:
          assert len(fw) == fw_size, f"{platform} {ecu}: {len(fw)} {fw_size}"


class TestSubaruLkasReleaseOnCameraEdge(unittest.TestCase):
  # EyeSight faults if the EPS keeps steering when the camera drops lane keep (SET_3 2->3) while ACC is off
  def _steer_request(self, ci, cc):
    while True:
      _, msgs = ci.CC.update(cc.as_reader(), CarControlSP(), ci.CS, 0)
      for addr, data, _ in msgs:
        if addr == 0x124:
          return (data[1] >> 4) & 1

  def test_release_on_low_speed_edge_only_without_acc(self):
    from opendbc.car.subaru.carcontroller import LKAS_RELEASE_FRAMES
    for acc_enabled in (False, True):
      with self.subTest(acc_enabled=acc_enabled):
        cp = CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023)
        ci = CarInterface(cp, CarInterface.get_non_essential_params_sp(cp, CAR.SUBARU_OUTBACK_2023))
        ci.update([])
        ci.CS.out = structs.CarState(vEgo=15, vEgoRaw=15)
        ci.CS.out.cruiseState.enabled = acc_enabled
        cc = structs.CarControl(latActive=True)
        ci.CS.cam_lkas_mode = 2
        self.assertEqual([self._steer_request(ci, cc) for _ in range(3)], [1, 1, 1])
        ci.CS.cam_lkas_mode = 3
        reqs = [self._steer_request(ci, cc) for _ in range(LKAS_RELEASE_FRAMES + 3)]
        if acc_enabled:
          self.assertEqual(reqs, [1] * (LKAS_RELEASE_FRAMES + 3))
        else:
          self.assertEqual(reqs, [0] * LKAS_RELEASE_FRAMES + [1, 1, 1])
        # staying in mode 3, or going back up to 2, never releases again
        ci.CS.cam_lkas_mode = 2
        self.assertEqual([self._steer_request(ci, cc) for _ in range(3)], [1, 1, 1])


class TestSubaruAngleLimits(unittest.TestCase):
  def setUp(self):
    cp = get_safety_CP()
    self.limits = CarControllerParams(cp)
    self.vm = VehicleModel(cp)

  def test_low_speed_deadband(self):
    for speed, desired, expected in ((3.9, 2.49, 0.0), (3.9, 2.5, 2.5), (3.9, -2.49, 0.0), (3.9, -2.5, -2.5), (4.0, 0.5, 0.5)):
      with self.subTest(speed=speed, desired=desired):
        cp = CarInterface.get_non_essential_params(CAR.SUBARU_CROSSTREK_2025)
        ci = CarInterface(cp, CarInterface.get_non_essential_params_sp(cp, CAR.SUBARU_CROSSTREK_2025))
        ci.update([])
        ci.CS.out = structs.CarState(vEgo=speed, vEgoRaw=speed)
        cc = structs.CarControl(latActive=True)
        cc.actuators.steeringAngleDeg = desired
        actuators, _ = ci.CC.update(cc.as_reader(), CarControlSP(), ci.CS, 0)
        self.assertAlmostEqual(actuators.steeringAngleDeg, expected)

  def test_safety_model_is_conservative(self):
    for platform in CAR:
      if not platform.config.flags & SubaruFlags.LKAS_ANGLE:
        continue
      vm = VehicleModel(CarInterface.get_non_essential_params(platform))
      for speed in np.linspace(1, 60, 120):
        with self.subTest(platform=platform, speed=speed):
          angle = min(get_max_angle_vm(speed, self.vm, self.limits), CarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX)
          accel = vm.calc_curvature(math.radians(angle), speed, 0) * speed ** 2
          self.assertLessEqual(accel, CarControllerParams.ANGLE_LIMITS.MAX_LATERAL_ACCEL + 1e-6)


class TestSubaruCruiseState(unittest.TestCase):
  def test_angle_cruise_uses_es_status(self):
    for platform in CAR:
      if not platform.config.flags & SubaruFlags.LKAS_ANGLE:
        continue
      cp = CarInterface.get_non_essential_params(platform)
      cp_sp = CarInterface.get_non_essential_params_sp(cp, platform)
      cs = CarState(cp, cp_sp)
      parsers = cs.get_can_parsers(cp, cp_sp)
      cruise_parser = parsers[Bus.alt if cp.flags & SubaruFlags.GLOBAL_GEN2 else Bus.cam]
      brake_parser = parsers[Bus.alt if cp.flags & SubaruFlags.GLOBAL_GEN2 else Bus.pt]
      for brake_pressed, status, brake_status in itertools.product((False, True), repeat=3):
        with self.subTest(platform=platform, brake=brake_pressed, status=status, es_brake=brake_status):
          brake_parser.vl["Brake_Status"]["Brake"] = brake_pressed
          cruise_parser.vl["ES_Status"]["Cruise_Activated"] = status
          cruise_parser.vl["ES_Brake"]["Cruise_Activated"] = brake_status
          self.assertEqual(cs.update(parsers)[0].cruiseState.enabled, status)

  def test_hybrid_cruise_uses_es_brake(self):
    for platform in CAR:
      if not platform.config.flags & SubaruFlags.HYBRID:
        continue
      cp = CarInterface.get_non_essential_params(platform)
      cp_sp = CarInterface.get_non_essential_params_sp(cp, platform)
      cs = CarState(cp, cp_sp)
      parsers = cs.get_can_parsers(cp, cp_sp)
      cruise_parser = parsers[Bus.alt if cp.flags & SubaruFlags.GLOBAL_GEN2 else Bus.cam]
      for enabled in (False, True):
        with self.subTest(platform=platform, enabled=enabled):
          cruise_parser.vl["ES_Brake"]["Cruise_Activated"] = enabled
          self.assertEqual(cs.update(parsers)[0].cruiseState.enabled, enabled)


class TestSubaruAvailability(unittest.TestCase):
  def test_angle_control_is_development_only(self):
    for platform in CAR:
      if not platform.config.flags & SubaruFlags.LKAS_ANGLE:
        continue
      for is_release in (False, True):
        with self.subTest(platform=platform, is_release=is_release):
          cp = CarInterface.get_params(platform, gen_empty_fingerprint(), [], False, is_release, False)
          self.assertEqual(cp.dashcamOnly, is_release or platform not in (CAR.SUBARU_CROSSTREK_2025, CAR.SUBARU_OUTBACK_2023))
