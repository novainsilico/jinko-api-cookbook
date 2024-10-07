.PHONY: all \
  login \
  test-ci

login:
	@{ \
		echo "Renew session login" 1>&2; \
		jk-basic-login; \
		status=$$?; \
		if [ $$status -ne 0 ] ; then \
			echo -e "\x1b[1;38;5;196m🠕 🠕 🠕 🠕 🠕 🠕 🠕 LOGIN ERROR 🠕 🠕 🠕 🠕 🠕 🠕 🠕\x1b[0m" 1>&2; \
			echo 1>&2; \
			echo "A login error can occur for the following reasons :" 1>&2; \
			echo 1>&2; \
			echo "- The environment has been stopped" 1>&2; \
			echo "- You need to join a trusted network" 1>&2; \
			echo "- The infrastructure has changed (IP whitelisting)" 1>&2; \
			echo 1>&2; \
		fi; \
		exit $$status; \
	}

test-ci: login
	@{ \
		cd ..; \
		./e2e/test/run_a_trial.sh || exit $$?; \
		echo done; \
	}